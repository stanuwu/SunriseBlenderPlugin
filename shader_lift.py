"""Lifts a scalar shader program to Blender-sized operations: vector math, clamps, mixes, dots and powers."""

# Operations the builder supplies; a constant with a TemplateLane is one too, linked to its native field.
LEAVES = frozenset({'constant', 'uv', 'screen', 'time', 'position', 'camera', 'view', 'surface', 'basis', 'frontface',
                    'sample', 'color', 'uvzw', 'instance', 'normal', 'rawposition', 'volume'})
# Leaves the builder also has as one vector socket, with how many lanes that socket holds.
VECTOR_LEAVES = {'sample': 3, 'surface': 3, 'basis': 3, 'position': 3, 'camera': 3, 'view': 3, 'normal': 3,
                 'color': 3, 'rawposition': 3, 'uv': 2, 'uvzw': 2}
# Program operations that one Math node computes.
SCALAR = {'add': 'ADD', 'mul': 'MULTIPLY', 'div': 'DIVIDE', 'mad': 'MULTIPLY_ADD', 'min': 'MINIMUM', 'max': 'MAXIMUM',
          'abs': 'ABSOLUTE', 'sqrt': 'SQRT', 'rsq': 'INVERSE_SQRT', 'frc': 'FRACT', 'round_ni': 'FLOOR',
          'round_ne': 'ROUND', 'round_pi': 'CEIL', 'round_z': 'TRUNC', 'sin': 'SINE', 'cos': 'COSINE', 'lt': 'LESS_THAN'}
# Program operations that one Vector Math node computes lane by lane.
VECTOR = {'add': 'ADD', 'mul': 'MULTIPLY', 'div': 'DIVIDE', 'mad': 'MULTIPLY_ADD', 'min': 'MINIMUM', 'max': 'MAXIMUM',
          'abs': 'ABSOLUTE', 'frc': 'FRACTION', 'round_ni': 'FLOOR', 'round_pi': 'CEIL', 'sin': 'SINE', 'cos': 'COSINE'}
# Lane operations a vector group may hold; neg is a scale, movc a mix and round_ne Blender's floor(x + 0.5).
GROUPABLE = frozenset(VECTOR) | {'neg', 'movc', 'round_ne'}
# Group operations that take two vector nodes.
TWO_NODES = frozenset({'round_ne'})
# Operand positions that may swap without changing the result.
SWAPS = {'add': (0, 1), 'mul': (0, 1), 'min': (0, 1), 'max': (0, 1), 'mad': (0, 1)}
# Lanes that mix these operations group as mad: mul(a, b) is mad(a, b, 0) and add(a, b) is mad(a, 1, b).
MAD_FAMILY = frozenset({'add', 'mul', 'mad'})
# Direct3D log of zero is -inf but Blender's is 0; values below the smallest normal float read -1e30.
LOG_FLOOR, LOG_ZERO = 1.1754944e-38, -1e30
# Constant refs that leave an add, a multiply or a negation's operand unchanged.
ZERO, ONE, MINUS = {('k', 0.), ('kv', (0., 0., 0.))}, {('k', 1.), ('kv', (1., 1., 1.))}, {('k', -1.), ('kv', (-1., -1., -1.))}


class Op:
    """One Blender node: kind is math, vector, mix, clamp or combine; inputs are refs.

    clamp is the 0 to 1 clamp of a Math node, or Clamp Result of an RGBA mix.
    """
    __slots__ = ('kind', 'operation', 'inputs', 'clamp', 'label')

    def __init__(self, kind, operation, inputs, label='', clamp=False):
        self.kind, self.operation, self.inputs, self.clamp, self.label = kind, operation, list(inputs), clamp, label


class Group:
    """Program nodes that are the lanes of one vector; operands are lane tuples, empty for a leaf."""
    __slots__ = ('lanes', 'operation', 'operands', 'padded')

    def __init__(self, lanes, operation, operands):
        self.lanes, self.operation, self.operands, self.padded = tuple(lanes), operation, operands, False


def is_vector(ref, ops):
    """Refs are ('k', value), ('kv', xyz), ('leaf', node), ('vleaf', lanes), ('op', index) and ('lane', ref, lane)."""
    if ref[0] in ('kv', 'vleaf'):
        return True
    if ref[0] == 'op':
        op = ops[ref[1]]
        return op.kind == 'combine' or op.kind == 'vector' and op.operation not in ('DOT_PRODUCT', 'LENGTH') \
            or op.kind == 'mix' and op.operation != 'FLOAT'
    return False


def _negated(ref):
    """-x for a constant ref."""
    return ('k', -ref[1]) if ref[0] == 'k' else ('kv', tuple(-value for value in ref[1]))


class Lift:
    """Lowers program nodes on demand; ops fill in the order an input precedes its user."""

    def __init__(self, program):
        # A copy, so the constants that pad mixed mad lanes can be appended.
        self.nodes = list(program['Nodes'])
        self.users = [[] for _ in self.nodes]
        for index, item in enumerate(self.nodes):
            for value in item['Inputs']:
                self.users[value].append(index)
        self.outputs = {index for indices in program['Outputs'].values() for index in indices}
        self.ops, self.memo, self.made, self.groups, self.group = [], {}, {}, [], {}
        # Plain groups first; padded mad groups then take only what is left.
        self.padding = False
        pending = self._leaf_groups()
        for padding in (False, True):
            self.padding = padding
            if padding:
                pending += range(len(self.groups))
            self._grow(pending)
            for indices in program['Outputs'].values():
                if len(indices) == 3:
                    self._top_down(tuple(indices), pending)
            for item in program['Nodes']:
                if item['Operation'] == 'sample':
                    self._top_down(tuple(item['Inputs'][:2]), pending)
            self._grow(pending)
        self._shared_seeds()
        self._unpad()

    # --- grouping ---

    def op(self, index):
        return self.nodes[index]['Operation']

    def inputs(self, index):
        return self.nodes[index]['Inputs']

    def uses(self, index):
        return len(self.users[index]) + (index in self.outputs)

    def plain(self, index):
        """A constant the builder writes as a value; a template lane links to its native field instead."""
        item = self.nodes[index]
        return item['Operation'] == 'constant' and item.get('TemplateLane') is None

    def _constant(self, value):
        """A plain constant node of this lift's own, for padding a lane."""
        key = ('const', value)
        if key not in self.memo:
            self.nodes.append({'Operation': 'constant', 'Inputs': [], 'Value': value})
            self.users.append([])
            self.memo[key] = len(self.nodes) - 1
        return self.memo[key]

    def _family(self, index):
        operation = self.op(index)
        return 'mad' if self.padding and operation in MAD_FAMILY else operation

    def _orders(self, index, operation):
        """The operand orders of one lane as the group's operation reads them."""
        own, values = self.op(index), tuple(self.inputs(index))
        if own == operation:
            orders = [values]
        elif own == 'mul':
            orders = [values + (self._constant(0.),)]
        else:
            one = self._constant(1.)
            orders = [(values[0], one, values[1]), (values[1], one, values[0])]
        swap = SWAPS.get(operation)
        if swap:
            for order in list(orders):
                swapped = list(order)
                swapped[swap[0]], swapped[swap[1]] = swapped[swap[1]], swapped[swap[0]]
                orders.append(tuple(swapped))
        return orders

    def _add_group(self, lanes, operation, operands, pending):
        self.groups.append(Group(lanes, operation, operands))
        for lane, index in enumerate(lanes):
            self.group[index] = (len(self.groups) - 1, lane)
        pending.append(len(self.groups) - 1)

    def _leaf_groups(self):
        """Leaf lanes that one socket holds, such as a texture's colour or a world vector."""
        found, pending = {}, []
        for index, item in enumerate(self.nodes):
            width = VECTOR_LEAVES.get(item['Operation'])
            if width is None:
                continue
            base = item['Channel'] // 2 * 2 if item['Operation'] == 'uv' else 0
            key = (item['Operation'], item.get('Slot'), item.get('SamplerSlot'), tuple(item['Inputs']), base)
            if item['Channel'] - base < width:
                found.setdefault(key, {})[item['Channel'] - base] = index
        for lanes in found.values():
            width = 0
            while width in lanes:
                width += 1
            if width >= 2:
                self._add_group([lanes[lane] for lane in range(width)], 'leaf', [], pending)
        return pending

    def classify(self, lanes):
        """How a tuple of program nodes enters a vector input: group, bcast, const or scalars."""
        if all(value == lanes[0] for value in lanes):
            return 'bcast'
        if all(self.plain(value) for value in lanes):
            return 'const'
        place = self.group.get(lanes[0])
        if place and place[1] == 0 and self.groups[place[0]].lanes[:len(lanes)] == tuple(lanes):
            return 'group'
        return 'scalars'

    def _try_group(self, lanes, pending, need_group):
        """Groups the lanes when one vector op computes them with at most one packed operand."""
        operations = {self.op(value) for value in lanes}
        operation = next(iter(operations)) if len(operations) == 1             else 'mad' if self.padding and operations <= MAD_FAMILY else None
        if operation not in GROUPABLE or len(set(lanes)) != len(lanes) or any(value in self.group for value in lanes):
            return False
        choices = [self._orders(value, operation) for value in lanes]
        best = None

        def search(lane, picked):
            nonlocal best
            if lane == len(lanes):
                operands = [tuple(order[position] for order in picked) for position in range(len(picked[0]))]
                kinds = [self.classify(value) for value in operands]
                score = (kinds.count('group'), -kinds.count('scalars'))
                if kinds.count('scalars') <= 1 and (not need_group or score[0]) and (best is None or score > best[0]):
                    best = (score, operands)
                return
            for order in choices[lane]:
                search(lane + 1, picked + [order])
        search(0, [])
        if best is None or self._cycles(lanes):
            return False
        self._add_group(lanes, operation, best[1], pending)
        self.groups[-1].padded = len(operations) > 1
        return True

    def _cycles(self, lanes):
        """True when one lane reads another, directly or through a group; one vector cannot compute both."""
        targets, seen = set(lanes), set()
        pending = [value for lane in lanes for value in self.inputs(lane)]
        while pending:
            value = pending.pop()
            if value in targets:
                return True
            if value in seen:
                continue
            seen.add(value)
            pending.extend(self.inputs(value))
            if value in self.group:
                pending.extend(self.groups[self.group[value][0]].lanes)
        return False

    def _shared_seeds(self):
        """Users that do the same thing to one shared value and constants become a group, such as waves of one time.

        The group and the groups grown from it stay only when they save nodes.
        """
        for index in range(len(self.nodes)):
            if self.plain(index):
                continue
            buckets = {}
            for user in sorted(set(self.users[index])):
                if user not in self.group and self.op(user) in GROUPABLE \
                        and all(value == index or self.plain(value) for value in self.inputs(user)):
                    buckets.setdefault(self._family(user), []).append(user)
            for users in buckets.values():
                for start in range(0, len(users) - 1, 3):
                    lanes = tuple(value for value in users[start:start + 3] if value not in self.group)
                    first, pending = len(self.groups), []
                    if len(lanes) < 2 or not self._try_group(lanes, pending, False):
                        continue
                    self._grow(pending)
                    if self._saving(first) <= 0:
                        self._drop(first)

    def _saving(self, first):
        """Nodes the groups from first on save: a lane each past the first, less a Combine or a Separate they need."""
        saving = 0
        for gid in range(first, len(self.groups)):
            group = self.groups[gid]
            if not group.lanes:
                continue
            saving += len(group.lanes) - 1 - (group.operation in TWO_NODES)
            saving -= sum(self.classify(operand) == 'scalars' for operand in group.operands)
            saving -= any(value in self.outputs or not self._read_as_vector(value, gid) for value in group.lanes)
        return saving

    def _drop(self, first):
        """Ungroups the groups from first on; a later group can only read these, so they all go."""
        for group in self.groups[first:]:
            for value in group.lanes:
                del self.group[value]
            group.lanes, group.padded = (), False

    def _unpad(self):
        """Drops each padded mad group whose lanes something reads one by one; a Separate would eat its saving."""
        changed = True
        while changed:
            changed = False
            for gid, group in enumerate(self.groups):
                if group.padded and any(value in self.outputs or not self._read_as_vector(value, gid)
                                        for value in group.lanes):
                    for value in group.lanes:
                        del self.group[value]
                    group.padded, group.lanes, changed = False, (), True

    def _read_as_vector(self, value, gid):
        """True when every reader of the lane reads the whole group as one vector operand."""
        lanes = self.groups[gid].lanes
        for user in self.users[value]:
            place = self.group.get(user)
            if place is None or not any(value in operand and tuple(operand) == lanes[:len(operand)]
                                        for operand in self.groups[place[0]].operands):
                return False
        return True

    def _grow(self, pending):
        """Bottom up: users of a group's lanes that do the same thing to each lane become a group."""
        while pending:
            group = self.groups[pending.pop()]
            for width in (len(group.lanes), 2):
                lanes = group.lanes[:width]
                found = False
                for first in self.users[lanes[0]]:
                    if first in self.group or self.op(first) not in GROUPABLE:
                        continue
                    # Same-operation lanes first, so a plain vector op wins over a padded mad.
                    candidates = [sorted((value for value in self.users[lane] if self._family(value) == self._family(first)
                                          and value not in self.group), key=lambda value: self.op(value) != self.op(first))
                                  for lane in lanes[1:]]
                    for rest in _product(candidates):
                        if self._try_group((first, *rest), pending, True):
                            found = True
                            break
                if found or len(group.lanes) == 2:
                    break

    def _top_down(self, lanes, pending):
        """Top down from an output: same-operation lanes group after their operands."""
        if len(lanes) < 2 or any(value in self.group or self.op(value) in LEAVES for value in lanes) \
                or len({self._family(value) for value in lanes}) != 1 or self.op(lanes[0]) not in GROUPABLE:
            return
        if len({self.op(value) for value in lanes}) != 1:
            # Mixed mad lanes group with their operands as they are.
            self._try_group(lanes, pending, False)
            return
        if len({len(self.inputs(value)) for value in lanes}) != 1:
            return
        for position in range(len(self.inputs(lanes[0]))):
            self._top_down(tuple(self.inputs(value)[position] for value in lanes), pending)
        self._try_group(lanes, pending, False)

    # --- lowering ---

    def add(self, kind, operation, *inputs, label='', clamp=False):
        """One op after the rewrites below; an op equal to one already made is that op.

        A rewrite only replaces this op. An input op it no longer reads is never built unless something else reads it.
        """
        simpler = self._simplify(kind, operation, inputs, label)
        if simpler is not None:
            return self._clamp(simpler) if clamp else simpler
        key = (kind, operation, tuple(inputs), clamp)
        if key not in self.made:
            self.ops.append(Op(kind, operation, inputs, label, clamp))
            self.made[key] = ('op', len(self.ops) - 1)
        return self.made[key]

    def _clamp(self, ref):
        """ref clamped to 0 to 1: the clamp option of its own Math node, else a Saturate node."""
        if ref[0] == 'op' and self.ops[ref[1]].kind == 'math':
            op = self.ops[ref[1]]
            return ref if op.clamp else self.add('math', op.operation, *op.inputs, label=op.label, clamp=True)
        return self.add('math', 'MAXIMUM', ref, ('k', 0.), label='Saturate', clamp=True)

    def _plain_op(self, ref, kind, *operations):
        """The unclamped op of this kind behind ref, when it is one of the operations; else None."""
        if ref[0] != 'op':
            return None
        op = self.ops[ref[1]]
        return op if op.kind == kind and op.operation in operations and not op.clamp else None

    def _negation(self, ref, kind):
        """x when ref is x * -1 in this kind of node, else None."""
        op = self._plain_op(ref, kind, 'MULTIPLY', 'SCALE')
        if op is None:
            return None
        if op.inputs[1] in MINUS:
            return op.inputs[0]
        return op.inputs[1] if op.operation == 'MULTIPLY' and op.inputs[0] in MINUS else None

    def _simplify(self, kind, operation, inputs, label):
        """An equal ref with fewer nodes, or None. Fused multiply-adds and a swapped mix factor round differently."""
        if kind == 'mix':
            # mix(a, b, 1 - t) is mix(b, a, t).
            factor = self._plain_op(inputs[0], 'math', 'SUBTRACT') or self._plain_op(inputs[0], 'vector', 'SUBTRACT')
            if operation != 'RGBA' and factor and factor.inputs[0] in ONE:
                return self.add(kind, operation, factor.inputs[1], inputs[2], inputs[1], label=label)
            return None
        if kind not in ('math', 'vector'):
            return None
        add = lambda *values: self.add(kind, *values)

        def keep(ref):
            """A vector node may become only another vector, and never a constant the caller cannot link."""
            if ref[0] in ('k', 'kv') or kind == 'vector' and not is_vector(ref, self.ops):
                return None
            return ref
        constant = lambda ref: ref[0] in ('k', 'kv')
        a = inputs[0]
        b = inputs[1] if len(inputs) > 1 else None
        if operation in ('ADD', 'SUBTRACT') and b in ZERO or operation in ('MULTIPLY', 'DIVIDE') and b in ONE \
                or operation == 'SCALE' and b == ('k', 1.):
            return keep(a)
        if operation in ('ADD', 'MULTIPLY') and a in ZERO | ONE and (a in ZERO) == (operation == 'ADD'):
            return keep(b)
        if operation == 'MULTIPLY_ADD':
            c = inputs[2]
            if a in ONE or b in ONE:
                return add('ADD', b if a in ONE else a, c)
            if c in ZERO:
                return add('MULTIPLY', a, b)
        # -(a - b) is b - a; -(x * k) is x * -k, and -(x * k + c) is x * -k - c.
        if operation in ('MULTIPLY', 'SCALE') and b in MINUS:
            x = self._plain_op(a, kind, 'MULTIPLY', 'MULTIPLY_ADD', 'SUBTRACT')
            if x and x.operation == 'SUBTRACT':
                return add('SUBTRACT', x.inputs[1], x.inputs[0])
            if x and (x.operation == 'MULTIPLY' or constant(x.inputs[2])):
                for side in (0, 1):
                    if constant(x.inputs[side]):
                        values = list(x.inputs)
                        values[side] = _negated(values[side])
                        values[2:] = map(_negated, values[2:])
                        return add(x.operation, *values)
        # -x * k is x * -k.
        if operation in ('MULTIPLY', 'MULTIPLY_ADD'):
            for this, other in ((0, 1), (1, 0)):
                inner = self._negation(inputs[this], kind)
                if inner is not None and constant(inputs[other]):
                    values = list(inputs)
                    values[this], values[other] = inner, _negated(inputs[other])
                    return add(operation, *values)
        if operation == 'ADD':
            # a + -b is a - b.
            for this, other in ((1, 0), (0, 1)):
                inner = self._negation(inputs[this], kind)
                if inner is not None:
                    return add('SUBTRACT', inputs[other], inner)
            # A multiply read by an add is one multiply-add.
            for this, other in ((0, 1), (1, 0)):
                product = self._plain_op(inputs[this], kind, 'MULTIPLY', 'SCALE')
                if product:
                    return add('MULTIPLY_ADD', *product.inputs, inputs[other])
        return None

    def scalar(self, index):
        """The value of one program node as a float ref."""
        key = ('s', index)
        if key not in self.memo:
            if self.plain(index):
                self.memo[key] = ('k', float(self.nodes[index]['Value']))
            elif self.op(index) in LEAVES:
                self.memo[key] = ('leaf', index)
            elif index in self.group:
                gid, lane = self.group[index]
                self.memo[key] = ('lane', self.vector(gid), lane)
            else:
                self.memo[key] = self._scalar_op(index)
        return self.memo[key]

    def vector(self, gid):
        """The value of one group as a vector ref."""
        key = ('v', gid)
        if key not in self.memo:
            group = self.groups[gid]
            self.memo[key] = ('vleaf', group.lanes) if group.operation == 'leaf' else self._vector_op(group)
        return self.memo[key]

    def operand(self, lanes):
        """A lane tuple as a vector ref; scalars are packed into one Combine XYZ."""
        kind = self.classify(lanes)
        if kind == 'bcast':
            value = self.scalar(lanes[0])
            return ('kv', (value[1],) * 3) if value[0] == 'k' else value
        if kind == 'const':
            # An unused lane repeats the last, so a constant of ones or zeros stays one for the rewrites.
            values = tuple(float(self.nodes[value]['Value']) for value in lanes)
            return ('kv', values + values[-1:] * (3 - len(values)))
        if kind == 'group':
            return self.vector(self.group[lanes[0]][0])
        key = ('c', lanes)
        if key not in self.memo:
            self.memo[key] = self.add('combine', '', *[self.scalar(value) for value in lanes], *[('k', 0.)] * (3 - len(lanes)))
        return self.memo[key]

    def as_vector(self, lanes):
        """A vector ref when the lanes already are one group, else None; a wider group's prefix counts."""
        lanes = tuple(lanes)
        return self.vector(self.group[lanes[0]][0]) if len(lanes) > 1 and self.classify(lanes) == 'group' else None

    def _negated(self, index):
        return self.op(index) == 'neg'

    def _lerp(self, lanes_operands):
        """Splits mad(b - a, t, a) per lane into (t, a, b) tuples; None for another shape."""
        for position in (0, 1):
            split = []
            for x, y, a in lanes_operands:
                d, t = (x, y) if position == 0 else (y, x)
                if self.op(d) != 'add':
                    break
                p, q = self.inputs(d)
                if self._negated(q) and self.inputs(q)[0] == a:
                    split.append((t, a, p))
                elif self._negated(p) and self.inputs(p)[0] == a:
                    split.append((t, a, q))
                else:
                    break
            else:
                return tuple(zip(*split))
        return None

    def _mask(self, condition):
        """A 0/1 factor for movc and whether it is inverted: (ref, inverted)."""
        operation = self.op(condition)
        if operation in ('lt', 'eq'):
            return self.scalar(condition), False
        if operation in ('ge', 'ne'):
            a, b = (self.scalar(value) for value in self.inputs(condition))
            return (self.add('math', 'LESS_THAN', a, b) if operation == 'ge' else self.add('math', 'COMPARE', a, b, ('k', 0.))), True
        return self.add('math', 'GREATER_THAN', self.add('math', 'ABSOLUTE', self.scalar(condition)), ('k', 0.)), False

    def _terms(self, index):
        """The summands of an add/mad tree: ('node', i) or ('prod', a, b); shared or grouped nodes stay whole."""
        terms = []

        def walk(value, top):
            operation = self.op(value)
            if not top and (value in self.group or self.uses(value) != 1 or operation not in ('add', 'mad', 'mul')):
                terms.append(('node', value))
            elif operation == 'add':
                walk(self.inputs(value)[0], False)
                walk(self.inputs(value)[1], False)
            elif operation == 'mad':
                terms.append(('prod', *self.inputs(value)[:2]))
                walk(self.inputs(value)[2], False)
            elif operation == 'mul':
                terms.append(('prod', *self.inputs(value)))
            else:
                terms.append(('node', value))
        walk(index, True)
        return terms

    def _dot(self, terms):
        """Finds three terms that are a 3-lane dot product: (a lanes, b lanes, remaining terms) or None."""
        # Summands that are the lanes of one product group.
        lanes = {}
        for term in terms:
            if term[0] == 'node' and term[1] in self.group:
                gid, lane = self.group[term[1]]
                if len(self.groups[gid].lanes) == 3:
                    lanes.setdefault(gid, {})[lane] = term
        for gid, found in lanes.items():
            if len(found) == 3:
                group = self.groups[gid]
                rest = [term for term in terms if term not in found.values()]
                if group.operation == 'mul':
                    return group.operands[0], group.operands[1], rest
                return group.lanes, (None,) * 3, rest
        # Products with one side in lane k of a 3-lane group.
        lanes = {}
        for term in terms:
            if term[0] == 'prod':
                for side in (1, 2):
                    place = self.group.get(term[side])
                    if place and len(self.groups[place[0]].lanes) == 3:
                        lanes.setdefault(place[0], {}).setdefault(place[1], (term, term[3 - side]))
        for gid, found in lanes.items():
            if len(found) == 3 and len({id(found[lane][0]) for lane in found}) == 3:
                rest = [term for term in terms if all(term is not found[lane][0] for lane in found)]
                return self.groups[gid].lanes, tuple(found[lane][1] for lane in range(3)), rest
        return None

    def _self_dot(self, index):
        """The lanes v when the node is exactly dot(v, v), else None."""
        if self.op(index) not in ('add', 'mad'):
            return None
        found = self._dot(self._terms(index))
        return found[0] if found and not found[2] and found[0] == found[1] else None

    def _scalar_op(self, index):
        operation, values = self.op(index), self.inputs(index)
        scalar = self.scalar
        if operation == 'min':
            for inner, high in (values, values[::-1]):
                if self.op(inner) == 'max' and self.uses(inner) == 1 and inner not in self.group:
                    x, low = self.inputs(inner)
                    if self.plain(x) and not self.plain(low):
                        x, low = low, x
                    if self._is(low, 0) and self._is(high, 1):
                        return self._saturate(x)
                    return self.add('clamp', 'MINMAX', scalar(x), scalar(low), scalar(high), label='Clamp')
        if operation == 'max':
            for inner, low in (values, values[::-1]):
                if self.op(inner) == 'min' and self.uses(inner) == 1 and inner not in self.group:
                    x, high = self.inputs(inner)
                    if self.plain(x) and not self.plain(high):
                        x, high = high, x
                    if self.plain(low) and self.plain(high) and self.nodes[low]['Value'] <= self.nodes[high]['Value']:
                        return self.add('clamp', 'MINMAX', scalar(x), scalar(low), scalar(high), label='Clamp')
        if operation in ('add', 'mad'):
            terms = self._terms(index)
            found = self._dot(terms) if len(terms) >= 3 else None
            if found:
                a, b, rest = found
                value = self.add('vector', 'DOT_PRODUCT', self.operand(a),
                                 ('kv', (1., 1., 1.)) if b == (None,) * 3 else self.operand(b), label='Dot')
                for term in rest:
                    value = self.add('math', 'MULTIPLY_ADD', scalar(term[1]), scalar(term[2]), value) if term[0] == 'prod' \
                        else self.add('math', 'ADD', value, scalar(term[1]))
                return value
        if operation == 'add':
            a, b = values
            if self._negated(b):
                return self.add('math', 'SUBTRACT', scalar(a), scalar(self.inputs(b)[0]))
            if self._negated(a):
                return self.add('math', 'SUBTRACT', scalar(b), scalar(self.inputs(a)[0]))
        if operation == 'mad':
            split = self._lerp([values])
            if split:
                (t,), (a,), (b,) = split
                return self.add('mix', 'FLOAT', scalar(t), scalar(a), scalar(b), label='Lerp')
        if operation == 'movc':
            mask, inverted = self._mask(values[0])
            a, b = scalar(values[1]), scalar(values[2])
            return self.add('mix', 'FLOAT', mask, *((a, b) if inverted else (b, a)), label='Select')
        if operation == 'exp':
            inner = values[0]
            if self.op(inner) == 'mul':
                for log, power in (self.inputs(inner), self.inputs(inner)[::-1]):
                    if self.op(log) == 'log':
                        return self.add('math', 'POWER', scalar(self.inputs(log)[0]), scalar(power), label='Pow')
            return self.add('math', 'POWER', ('k', 2.), scalar(inner))
        if operation == 'sqrt':
            lanes = self._self_dot(values[0])
            if lanes:
                return self.add('vector', 'LENGTH', self.operand(lanes), label='Length')
        if operation == 'neg':
            return self.add('math', 'MULTIPLY', scalar(values[0]), ('k', -1.))
        if operation == 'log':
            x = scalar(values[0])
            return self.add('math', 'MULTIPLY_ADD', self.add('math', 'LESS_THAN', x, ('k', LOG_FLOOR)), ('k', LOG_ZERO),
                            self.add('math', 'LOGARITHM', x, ('k', 2.)))
        if operation in ('eq', 'ne'):
            same = self.add('math', 'COMPARE', scalar(values[0]), scalar(values[1]), ('k', 0.))
            return self.add('math', 'SUBTRACT', ('k', 1.), same) if operation == 'ne' else same
        if operation == 'ge':
            return self.add('math', 'SUBTRACT', ('k', 1.), self.add('math', 'LESS_THAN', scalar(values[0]), scalar(values[1])))
        if operation in SCALAR:
            return self.add('math', SCALAR[operation], *[scalar(value) for value in values])
        raise ValueError('Unsupported shader expression: ' + operation)

    def _is(self, index, value):
        return self.plain(index) and self.nodes[index]['Value'] == value

    def _saturate(self, x):
        """saturate(x): the clamp option on a copy of x's Math node; the unclamped one is built only if read."""
        return self._clamp(self.scalar(x))

    def _vector_saturate(self, operation, operands):
        """min(max(v, 0), 1) or max(min(v, 1), 0) as one clamped mix, else None."""
        inner_operation, bound, inner_bound = ('max', 1., 0.) if operation == 'min' else ('min', 0., 1.)
        for position in (0, 1):
            inner, limit = operands[position], operands[1 - position]
            if not all(self._is(value, bound) for value in limit) or self.classify(inner) != 'group':
                continue
            group = self.groups[self.group[inner[0]][0]]
            if group.operation != inner_operation:
                continue
            for side in (0, 1):
                if all(self._is(value, inner_bound) for value in group.operands[1 - side]):
                    return self.add('mix', 'RGBA', ('k', 0.), self.operand(group.operands[side]), ('k', 0.),
                                    label='Saturate', clamp=True)
        return None

    def _vector_op(self, group):
        operation, operands = group.operation, group.operands
        if operation in ('min', 'max'):
            value = self._vector_saturate(operation, operands)
            if value is not None:
                return value
        if operation == 'neg':
            return self.add('vector', 'SCALE', self.operand(operands[0]), ('k', -1.))
        if operation == 'add':
            for position in (1, 0):
                if all(self._negated(value) for value in operands[position]):
                    inner = tuple(self.inputs(value)[0] for value in operands[position])
                    return self.add('vector', 'SUBTRACT', self.operand(operands[1 - position]), self.operand(inner))
        if operation == 'mul':
            for position in (0, 1):
                rate = operands[1 - position]
                if self.classify(rate) == 'bcast' and self.op(rate[0]) == 'rsq' \
                        and self._self_dot(self.inputs(rate[0])[0]) == operands[position]:
                    return self.add('vector', 'NORMALIZE', self.operand(operands[position]), label='Normalize')
        if operation == 'mad':
            split = self._lerp(list(zip(*operands)))
            if split:
                t, a, b = split
                factor = self.scalar(t[0]) if self.classify(t) == 'bcast' else self.operand(t)
                return self.add('mix', 'VECTOR', factor, self.operand(a), self.operand(b), label='Lerp')
        if operation == 'movc':
            conditions, a, b = operands
            if self.classify(conditions) == 'bcast':
                mask, inverted = self._mask(conditions[0])
            else:
                masks = [self._mask(value) for value in conditions]
                inverted = masks[0][1]
                if any(value[1] != inverted for value in masks):
                    masks = [(self.add('math', 'SUBTRACT', ('k', 1.), ref), False) if flip else (ref, False) for ref, flip in masks]
                    inverted = False
                mask = self.add('combine', '', *[ref for ref, _ in masks], *[('k', 0.)] * (3 - len(masks)))
            a, b = self.operand(a), self.operand(b)
            return self.add('mix', 'VECTOR', mask, *((a, b) if inverted else (b, a)), label='Select')
        if operation == 'round_ne':
            return self.add('vector', 'FLOOR', self.add('vector', 'ADD', self.operand(operands[0]), ('kv', (.5, .5, .5))))
        return self.add('vector', VECTOR[operation], *[self.operand(value) for value in operands])


def _product(lists):
    if not lists:
        yield ()
        return
    for head in lists[0]:
        for rest in _product(lists[1:]):
            yield (head, *rest)

