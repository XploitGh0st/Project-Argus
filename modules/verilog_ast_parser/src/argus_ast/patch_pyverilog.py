"""
Consolidated runtime patches for Pyverilog bugs and performance bottlenecks
encountered on real Trust-Hub benchmark circuits.
"""

import sys
import copy
import re

import pyverilog.utils.verror as verror
from pyverilog.utils.scope import ScopeChain, ScopeLabel
import pyverilog.utils.signaltype as signaltype
import pyverilog.vparser.ast as vast
from pyverilog.dataflow.dataflow import DFIntConst, Term, Bind
import pyverilog.dataflow.signalvisitor as sv
from pyverilog.dataflow.signalvisitor import SignalVisitor
from pyverilog.dataflow.bindvisitor import BindVisitor
from pyverilog.dataflow.optimizer import VerilogOptimizer, DFEvalValue
import pyverilog.dataflow.merge as df_merge
from pyverilog.controlflow.controlflow_analyzer import VerilogControlflowAnalyzer


_PATCHED = False


def apply_pyverilog_patches():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # 1. ScopeLabel & ScopeChain fast hashing and slicing (avoids deepcopy and tuple conversion overhead)
    def _scope_label_hash(self):
        if not hasattr(self, "_cached_hash"):
            self._cached_hash = hash((self.scopename, self.scopeloop))
        return self._cached_hash

    def _scope_label_eq(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return (
            self.scopename == other.scopename
            and self.scopeloop == other.scopeloop
            and self.scopetype == other.scopetype
        )

    ScopeLabel.__hash__ = _scope_label_hash
    ScopeLabel.__eq__ = _scope_label_eq

    def _scopechain_init(self, scopechain=None):
        if scopechain is None:
            self.scopechain = ()
        elif isinstance(scopechain, tuple):
            self.scopechain = scopechain
        else:
            self.scopechain = tuple(scopechain)
        self._cached_hash = hash(self.scopechain)

    def _scopechain_add(self, r):
        if isinstance(r, ScopeLabel):
            return ScopeChain(self.scopechain + (r,))
        elif isinstance(r, ScopeChain):
            return ScopeChain(self.scopechain + r.scopechain)
        raise verror.DefinitionError(f"Can not add {r}")

    def _scopechain_getitem(self, key):
        if isinstance(key, slice):
            return ScopeChain(self.scopechain[key])
        return self.scopechain[key]

    def _scopechain_hash(self):
        return self._cached_hash

    def _scopechain_eq(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        if self._cached_hash != other._cached_hash or len(self.scopechain) != len(other.scopechain):
            return False
        return self.scopechain == other.scopechain

    ScopeChain.__init__ = _scopechain_init
    ScopeChain.__add__ = _scopechain_add
    ScopeChain.__getitem__ = _scopechain_getitem
    ScopeChain.__hash__ = _scopechain_hash
    ScopeChain.__eq__ = _scopechain_eq

    # 2. Term shallow copy in getRenamedDst (replaces deepcopy)
    def _patched_getRenamedDst(self, dst):
        renamed_dst = ()
        for d in dst:
            dname = d[0]
            while True:
                renamed_dname = self.renameVar(dname)
                if not self.dataflow.hasTerm(renamed_dname):
                    break
            term = self.dataflow.getTerm(dname)
            newterm = Term(renamed_dname, set(["Rename"]), term.msb, term.lsb)
            self.dataflow.addTerm(renamed_dname, newterm)
            renamed_dst += ((renamed_dname,) + d[1:],)
        return renamed_dst

    BindVisitor.getRenamedDst = _patched_getRenamedDst

    # 3. Finite loop in setInstanceConstants & copyFrameInfo (breaks when no forward progress, skips non-module scopes)
    def _patched_setInstanceConstants(self):
        current = self.frames.getCurrent()
        if not self.frames.dict[current].isModule():
            return
        while True:
            all_passed, progress = True, False
            for name, definitions in self.frames.getConsts(current).items():
                if len(definitions) > 1 or self.hasConstant(name):
                    continue
                for definition in definitions:
                    if getattr(definition, "__class__", None).__name__ == "Genvar":
                        continue
                    try:
                        val = self.optimize(self.getTree(definition.value, current))
                        if isinstance(val, DFEvalValue):
                            self.setConstant(name, val)
                            progress = True
                        else:
                            all_passed = False
                    except Exception:
                        all_passed = False
            if all_passed or not progress:
                break

    SignalVisitor.setInstanceConstants = _patched_setInstanceConstants

    def _patched_copyFrameInfo(self, current):
        if not self.frames.dict[current].isModule():
            return
        for name, definitions in self.frames.getConsts(current).items():
            if len(definitions) > 1:
                continue
            for definition in definitions:
                self.dataflow.addTerm(name, sv.Term(name, {definition.__class__.__name__}))
                self.setConstantTerm(name, self.makeConstantTerm(name, definition, current))
        while True:
            all_passed, progress = True, False
            for name, definitions in self.frames.getConsts(current).items():
                if len(definitions) > 1 or self.hasConstant(name):
                    continue
                for definition in definitions:
                    if getattr(definition, "__class__", None).__name__ == "Genvar":
                        continue
                    try:
                        val = self.optimize(self.getTree(definition.value, current))
                        if isinstance(val, DFEvalValue):
                            self.setConstant(name, val)
                            progress = True
                        else:
                            all_passed = False
                    except Exception:
                        all_passed = False
            if all_passed or not progress:
                break

    BindVisitor.copyFrameInfo = _patched_copyFrameInfo

    # 4. makeConstantTerm width None guard
    def _patched_makeConstantTerm(self, name, node, scope):
        msb = (
            DFIntConst("31")
            if (node.width is None or getattr(node.width, "msb", None) is None)
            else self.makeDFTree(node.width.msb, scope)
        )
        lsb = (
            DFIntConst("0")
            if (node.width is None or getattr(node.width, "lsb", None) is None)
            else self.makeDFTree(node.width.lsb, scope)
        )
        return Term(name, {node.__class__.__name__}, msb, lsb)

    SignalVisitor.makeConstantTerm = _patched_makeConstantTerm
    BindVisitor.makeConstantTerm = _patched_makeConstantTerm

    # 5. DFUndefined safe optimizeConstant
    orig_optConst = VerilogOptimizer.optimizeConstant

    def _patched_optConst(self, tree):
        if hasattr(tree, "name") and self.hasConstant(tree.name):
            c = self.getConstant(tree.name)
            if not hasattr(c, "value"):
                return tree
        try:
            return orig_optConst(self, tree)
        except AttributeError:
            return tree

    VerilogOptimizer.optimizeConstant = _patched_optConst

    # 6. Sensitivity list clock edge fallback
    def _patched_createAlwaysinfo(self, node, current):
        scope = current
        clock_name, clock_edge, clock_bit = None, None, 0
        reset_name, reset_edge, reset_bit = None, None, 0
        senslist = []
        for l in node.sens_list.list:
            if l.type == "level":
                senslist.append(l)
                continue
            if isinstance(l.sig, vast.Pointer):
                signame = self._get_signal_name(l.sig.var)
                bit = int(l.sig.ptr.value)
            else:
                signame = self._get_signal_name(l.sig)
                bit = 0
            if signame and signaltype.isClock(signame):
                clock_name, clock_edge, clock_bit = self.searchTerminal(signame, scope), l.type, bit
            elif signame and signaltype.isReset(signame):
                reset_name, reset_edge, reset_bit = self.searchTerminal(signame, scope), l.type, bit
            elif clock_edge is None and signame:
                clock_name, clock_edge, clock_bit = self.searchTerminal(signame, scope), l.type, bit
            elif reset_edge is None and signame:
                reset_name, reset_edge, reset_bit = self.searchTerminal(signame, scope), l.type, bit
            else:
                senslist.append(l)
        return (clock_name, clock_edge, clock_bit, reset_name, reset_edge, reset_bit, senslist)

    BindVisitor._createAlwaysinfo = _patched_createAlwaysinfo

    # 7. isFsmVar candidate filter (avoids analyzing 128-bit AES arrays or SSA renames as FSMs)
    orig_isFsmVar = VerilogControlflowAnalyzer.isFsmVar

    def _patched_isFsmVar(self, termname):
        name = str(termname)
        if "_rn" in name or "next" in name.lower():
            return False
        try:
            w = self.getWidth(termname)
            if w is not None and w > 8:
                return False
        except Exception:
            pass
        return orig_isFsmVar(self, termname)

    VerilogControlflowAnalyzer.isFsmVar = _patched_isFsmVar

    # 8. Bind fast shallow clone (replaces recursive copy.deepcopy of huge trees)
    def _bind_deepcopy(self, memo):
        return Bind(self.tree, self.dest, self.msb, self.lsb, self.ptr, self.alwaysinfo, self.parameterinfo)

    Bind.__deepcopy__ = _bind_deepcopy

    # 9. BindVisitor.searchTerminal cache & fast if-scope skip
    orig_searchTerminal = BindVisitor.searchTerminal

    def _patched_searchTerminal(self, name, scope):
        if not hasattr(self, "_search_cache"):
            self._search_cache = {}
        key = (name, scope)
        if key in self._search_cache:
            return self._search_cache[key]
        idx = len(scope.scopechain)
        while idx > 0 and scope.scopechain[idx - 1].scopetype == "if":
            idx -= 1
        s = scope[:idx] if idx < len(scope.scopechain) else scope
        res = orig_searchTerminal(self, name, s)
        self._search_cache[key] = res
        return res

    BindVisitor.searchTerminal = _patched_searchTerminal

    # 10. getOptimizedBindlist fast Bind construction
    def _patched_getOptimizedBindlist(self, bindlist):
        if len(bindlist) == 0:
            return ()
        new_bindlist = []
        for bind in bindlist:
            tree = self.optimizer.optimize(bind.tree)
            msb = self.optimizer.optimize(bind.msb)
            lsb = self.optimizer.optimize(bind.lsb)
            ptr = self.optimizer.optimize(bind.ptr)
            new_bind = Bind(tree, bind.dest, msb, lsb, ptr, bind.alwaysinfo, bind.parameterinfo)
            new_bindlist.append(new_bind)
        if len(new_bindlist) == 1:
            return (new_bindlist[0],)
        split_positions = self.splitPositions(tuple(new_bindlist))
        new_bindlist = self.splitBindlist(tuple(new_bindlist), split_positions)
        return self.mergeBindlist(tuple(new_bindlist))

    df_merge.VerilogDataflowMerge.getOptimizedBindlist = _patched_getOptimizedBindlist

    # 11. FrameTable blocking assign indexing (O(1) exit when target signal has no blocking assignments)
    import pyverilog.dataflow.visit as df_visit

    orig_setBlockingAssign = df_visit.FrameTable.setBlockingAssign
    def _patched_setBlockingAssign(self, dst, bind, scope):
        if not hasattr(self, "_blocking_dsts"):
            self._blocking_dsts = set()
        self._blocking_dsts.add(dst)
        orig_setBlockingAssign(self, dst, bind, scope)

    def _patched_getBlockingAssign(self, dst, scope):
        if not hasattr(self, "_blocking_dsts") or dst not in self._blocking_dsts:
            return ()
        p = scope
        while self.dict[p].isAlways():
            ret = self.dict[p].getBlockingAssign(dst)
            if ret:
                return ret
            p = self.dict[p].getPrevious()
        return ()

    df_visit.FrameTable.setBlockingAssign = _patched_setBlockingAssign
    df_visit.FrameTable.getBlockingAssign = _patched_getBlockingAssign

    # 12. Fast getCondlist & getFlowlist dynamic programming
    def _patched_getCondlist(self, scope):
        if not hasattr(self, "_condlist_cache"):
            self._condlist_cache = {}
        if scope in self._condlist_cache:
            return self._condlist_cache[scope]
        frame = self.frames.dict[scope]
        if frame.isModule() or frame.previous is None:
            parent_condlist = ()
        else:
            parent_condlist = self.getCondlist(frame.previous)
        cond = frame.getCondition()
        if cond is not None:
            tree = self.makeDFTree(cond, self.reduceIfScope(scope))
            res = parent_condlist + (tree,)
        else:
            res = parent_condlist
        self._condlist_cache[scope] = res
        return res

    def _patched_getFlowlist(self, scope):
        if not hasattr(self, "_flowlist_cache"):
            self._flowlist_cache = {}
        if scope in self._flowlist_cache:
            return self._flowlist_cache[scope]
        frame = self.frames.dict[scope]
        if frame.isModule() or frame.previous is None:
            parent_flowlist = ()
        else:
            parent_flowlist = self.getFlowlist(frame.previous)
        cond = frame.getCondition()
        if cond is not None:
            res = parent_flowlist + (not frame.isIfelse(),)
        else:
            res = parent_flowlist
        self._flowlist_cache[scope] = res
        return res

    orig_resolveCondlist = BindVisitor.resolveCondlist
    def _patched_resolveCondlist(self, condlist, scope):
        if not hasattr(self.frames, "_blocking_dsts") or len(self.frames._blocking_dsts) == 0:
            return condlist
        return orig_resolveCondlist(self, condlist, scope)

    # 14. Fast BindVisitor.makeBind (skips expensive O(N^2) reorder when raw_tree has no nested branch)
    import pyverilog.dataflow.reorder as df_reorder

    from pyverilog.dataflow.dataflow import DFBranch, DFPartselect

    def _has_branch(node):
        if node is None or isinstance(node, (DFIntConst, DFEvalValue)):
            return False
        if isinstance(node, DFBranch):
            return True
        if hasattr(node, "nextnodes"):
            return any(_has_branch(n) for n in node.nextnodes)
        return False

    orig_makeBind = BindVisitor.makeBind
    def _patched_makeBind(self, name, msb, lsb, ptr, part_msb, part_lsb,
                          raw_tree, condlist, flowlist,
                          num_dst=1, alwaysinfo=None, bindtype=None):
        current_bindlist = self.getBindlist(name)
        current_tree = None
        current_msb = None
        current_lsb = None
        current_ptr = None

        if len(current_bindlist) > 0:
            for current_bind in current_bindlist:
                if (current_bind.msb == msb and
                    current_bind.lsb == lsb and
                    current_bind.ptr == ptr):
                    current_tree = current_bind.tree
                    current_msb = current_bind.msb
                    current_lsb = current_bind.lsb
                    current_ptr = current_bind.ptr
                    break

        rest_tree = current_tree
        rest_condlist = condlist
        rest_flowlist = flowlist

        match_flowlist = ()
        if (current_msb == msb and
            current_lsb == lsb and
            current_ptr == ptr):
            (rest_tree,
             rest_condlist,
             rest_flowlist,
             match_flowlist) = self.diffBranchTree(current_tree, condlist, flowlist)

        add_tree = self.makeBranchTree(rest_condlist, rest_flowlist, raw_tree)
        if rest_flowlist and rest_tree is not None:
            _rest_flowlist = rest_flowlist[:-1] + (not rest_flowlist[-1],)
            add_tree = self.appendBranchTree(add_tree, _rest_flowlist, rest_tree)

        tree = self.appendBranchTree(current_tree, match_flowlist, add_tree)
        if _has_branch(raw_tree):
            tree = df_reorder.reorder(tree)

        if num_dst > 1:
            tree = df_reorder.reorder(
                DFPartselect(tree, part_msb, part_lsb)
            )

        return Bind(tree, name, msb, lsb, ptr, alwaysinfo, bindtype)

    BindVisitor.makeBind = _patched_makeBind

    # 15. FrameTable getAlwaysStatus caching
    orig_getAlwaysStatus = df_visit.FrameTable.getAlwaysStatus
    def _patched_getAlwaysStatus(self):
        cur = self.current
        frame = self.dict[cur]
        if hasattr(frame, "_cached_always_status"):
            return frame._cached_always_status
        res = orig_getAlwaysStatus(self)
        frame._cached_always_status = res
        return res
    df_visit.FrameTable.getAlwaysStatus = _patched_getAlwaysStatus

    # 16. Flat case statement handling (for wide cases like SBox > 32 items to prevent deep recursion)
    from pyverilog.vparser.ast import Eq, Lor, IntConst

    orig_bindvisitor_case = BindVisitor._case

    def _patched_case(self, comp, caselist, myframes):
        if len(caselist) <= 32:
            return orig_bindvisitor_case(self, comp, caselist, myframes)
        start_frame = self.frames.getCurrent()
        for case in caselist:
            cond = IntConst('1')
            if case.cond is not None:
                if len(case.cond) > 1:
                    cond = Eq(comp, case.cond[0])
                    for c in case.cond[1:]:
                        cond = Lor(cond, Eq(comp, c))
                else:
                    cond = Eq(comp, case.cond[0])
            label = self.labels.get(self.frames.getLabelKey('if'))
            current = self.stackNextFrame(label, 'if',
                                          frametype='ifthen',
                                          condition=cond,
                                          functioncall=self.frames.isFunctioncall(),
                                          taskcall=self.frames.isTaskcall(),
                                          generate=self.frames.isGenerate(),
                                          always=self.frames.isAlways(),
                                          initial=self.frames.isInitial())
            myframes.append(self.frames.getCurrent())
            if case.statement is not None:
                self.visit(case.statement)
            self.frames.setCurrent(start_frame)

    BindVisitor._case = _patched_case
    orig_getDsts = BindVisitor.getDsts
    def _patched_getDsts(self, left, scope):
        if isinstance(left, vast.Lvalue):
            return self.getDsts(left.var, scope)
        if isinstance(left, (vast.LConcat, vast.Concat)):
            dst = []
            for n in left.list:
                dst.extend(list(self.getDsts(n, scope)))
            return tuple(dst)
        return orig_getDsts(self, left, scope)
    BindVisitor.getDsts = _patched_getDsts

    BindVisitor.getCondlist = _patched_getCondlist
    BindVisitor.getFlowlist = _patched_getFlowlist
    BindVisitor.resolveCondlist = _patched_resolveCondlist
