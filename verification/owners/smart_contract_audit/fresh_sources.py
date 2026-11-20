# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Original self-contained source pathways, not copies of an exposed audit.

Every source uses only the explicit caller, arguments and zero-initialized local
storage. No chain, transaction origin, timestamp or external contract is needed.
Training additions and reserved audit designs are separate implementations.
"""


def make(entity, body, facts, prefix=""):
    source = "// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\npragma solidity 0.8.30;\n" + prefix + f"contract C{entity}{{{body}}}\n"
    facts["contract"] = f"C{entity}"
    return source, facts


def training(index, entity, vulnerable):
    if not 0 <= index < 12:
        raise ValueError("bounded original training design index required")
    f, rd, owner = f"f{entity}", f"read{entity}", 100 + entity
    if index < 6:
        guard = "require(msg.sender!=address(0));" if vulnerable else f"require(msg.sender==address({owner}));"
        if index == 0:
            predicate = "iszero(caller())" if vulnerable else f"iszero(eq(caller(),{owner}))"
            guard = f"assembly{{if {predicate}{{revert(0,0)}}}}"
            store, write, read = "uint256[2] values;", "values[1]=x;", "values[1]"
        elif index == 1:
            guard = f"bool permission=msg.sender==address({owner});require(permission" + ("||x>0" if vulnerable else "") + ");"
            store, write, read = "mapping(address=>uint256) values;", f"values[address({owner})]=x;", f"values[address({owner})]"
        elif index == 2:
            guard = f"uint160 delta=uint160(msg.sender)^uint160({owner});require(" + ("delta!=type(uint160).max" if vulnerable else "delta==0") + ");"
            store, write, read = "struct Box{uint256 value;} Box box;", "box.value=x;", "box.value"
        elif index == 3:
            guard = f"if(msg.sender!=address({owner})){{" + ("require(x>0);" if vulnerable else "revert();") + "}"
            store, write, read = "uint256 value;", "value=x;", "value"
        elif index == 4:
            guard = f"address[1] memory permitted=[address({owner})];require(" + ("permitted[0]!=address(0)" if vulnerable else "msg.sender==permitted[0]") + ");"
            store, write, read = "bytes32 value;", "value=bytes32(x);", "uint256(value)"
        else:
            guard = f"require(uint256(uint160(msg.sender))" + (">0" if vulnerable else f"=={owner}") + ");"
            store, write, read = "mapping(uint256=>uint256) values;", "values[x%3]=x;last=x%3;", "values[last]"
            store += "uint256 last;"
        body = store + f"function {f}(uint256 x)external{{{guard}{write}}}function {rd}()external view returns(uint256){{return {read};}}"
        facts = {"mechanism": "access_control", "function": f, "signature": f + "(uint256)", "read_function": rd,
                 "owner": owner, "attacker": owner + 10000, "argument": 17, "bad_args": [17], "normal_args": [18]}
        return make(entity, body, facts)
    width = (16, 24, 32)[index % 3]
    typ, maximum = f"uint{width}", (1 << width) - 1
    expressions = ("(x+y)*3", "x*x+y", "x*(y+1)", "(x+1)*(y+1)", "x+x+y", "(x*y)+x")
    expression = expressions[index - 6]
    bound = f"type({typ}).max"
    guards = (
        f"require(x<={bound}-y);require(x+y<={bound}/3);",
        f"require(x==0||x<={bound}/x);require(x*x<={bound}-y);",
        f"require(y<{bound});require(x<={bound}/(y+1));",
        f"require(x<{bound}&&y<{bound});require(x+1<={bound}/(y+1));",
        f"require(x<={bound}/2);require(x+x<={bound}-y);",
        f"require(y==0||x<={bound}/y);require(x*y<={bound}-x);",
    )
    checked = ("" if vulnerable else guards[index - 6]) + f"unchecked{{return {expression};}}"
    # A memory tuple and distinct assertion separate these training pathways
    # from the reserved source-held-out implementation shapes.
    body = f"struct Pair{{{typ} a;{typ} b;}}function {f}({typ} x,{typ} y)external pure returns({typ}){{Pair memory p=Pair(x,y);require(p.a==x&&p.b==y);{checked}}}"
    wrapped = eval(expression, {"__builtins__": {}}, {"x": maximum, "y": 2}) % (1 << width)
    normal = eval(expression, {"__builtins__": {}}, {"x": 6, "y": 2})
    facts = {"mechanism": "arithmetic", "function": f, "signature": f + f"({typ},{typ})", "width": width,
             "bad_args": [maximum, 2], "good_args": [6, 2], "wrapped_result": wrapped, "normal_result": normal}
    return make(entity, body, facts)


def audit(index, entity, vulnerable):
    if not 0 <= index < 32:
        raise ValueError("bounded clean audit design index required")
    f, rd, owner = f"f{entity}", f"read{entity}", 100 + entity
    if index < 16:
        open_guard = "require(msg.sender!=address(0));"
        closed_guard = f"require(msg.sender==address({owner}));"
        guard, extra, suffix, prefix = open_guard if vulnerable else closed_guard, "", "", ""
        stores = ("uint256 value;", "uint256[2] values;", "struct Box{uint256 amount;} Box box;", "mapping(uint256=>uint256) values;")
        writes = ("value=x;", "values[0]=x;", "box.amount=x;", "values[7]=x;")
        reads = ("value", "values[0]", "box.amount", "values[7]")
        if index == 0:
            condition, rejected = ("iszero(caller())", 1) if vulnerable else (f"eq(caller(),{owner})", 0)
            guard = f"assembly{{switch {condition} case {rejected}{{revert(0,0)}}}}"
        elif index == 1:
            guard = f"uint160 d=uint160(msg.sender)|uint160({owner});require(" + ("d!=0" if vulnerable else f"d=={owner}&&uint160(msg.sender)=={owner}") + ");"
        elif index == 2:
            guard = f"bool a=msg.sender==address({owner});bool b=x>0;require(" + ("a||b" if vulnerable else "a&&b") + ");"
        elif index == 3:
            extra = f"modifier only{entity}(){{{guard}_;}}"
            guard, suffix = "", f" only{entity}"
        elif index == 4:
            extra = f"function auth{entity}(address a)private pure returns(bool){{return a==address({owner});}}"
            guard = f"require(auth{entity}(" + (f"address({owner})" if vulnerable else "msg.sender") + "));"
        elif index == 5:
            extra = "error Denied();"
            guard = "if(msg.sender==address(0))revert Denied();" if vulnerable else f"if(msg.sender!=address({owner}))revert Denied();"
        elif index == 6:
            guard = "uint160 u=uint160(msg.sender);require(" + ("u!=0" if vulnerable else f"u/1=={owner}") + ");"
        elif index == 7:
            extra = "enum Rank{visitor,holder}"
            guard = f"Rank r=msg.sender==address({owner})?Rank.holder:Rank.visitor;require(" + ("r==Rank.visitor||r==Rank.holder" if vulnerable else "r==Rank.holder") + ");"
        elif index == 8:
            guard = f"address[2] memory a=[address({owner}),address(0)];bool ok=false;for(uint256 i=0;i<2;i++){{if(" + ("msg.sender!=a[i]" if vulnerable else "msg.sender==a[i]") + ")ok=true;}require(ok);"
        elif index == 9:
            guard = "int256 u=int256(uint256(uint160(msg.sender)));require(" + ("u>0" if vulnerable else f"u=={owner}") + ");"
        elif index == 10:
            extra = "struct Authority{address holder;}"
            guard = f"Authority memory a=Authority(address({owner}));require(" + ("a.holder!=address(0)" if vulnerable else "msg.sender==a.holder") + ");"
        elif index == 11:
            guard = "require(bytes20(msg.sender)" + ("!=bytes20(address(0))" if vulnerable else f"==bytes20(address({owner}))") + ");"
        elif index == 12:
            prefix = "type Identity is uint160;"
            guard = "Identity a=Identity.wrap(uint160(msg.sender));require(Identity.unwrap(a)" + (">0" if vulnerable else f"=={owner}") + ");"
        elif index == 13:
            guard = f"uint160 u=uint160(msg.sender);uint160 d=(u&uint160({owner}))|(u^uint160({owner}));require(" + ("d!=0" if vulnerable else f"d=={owner}&&u=={owner}") + ");"
        elif index == 14:
            guard = "assert(msg.sender!=address(0));" if vulnerable else f"assert(msg.sender==address({owner}));"
        elif index == 15:
            guard = "if(x>0){" + ("require(msg.sender!=address(0));" if vulnerable else f"if(msg.sender!=address({owner}))revert();") + "}"
        which = (index + 1) % 4
        body = stores[which] + extra + f"function {f}(uint256 x)external{suffix}{{{guard}{writes[which]}}}function {rd}()external view returns(uint256){{return {reads[which]};}}"
        facts = {"mechanism": "access_control", "function": f, "signature": f + "(uint256)", "read_function": rd,
                 "owner": owner, "attacker": owner + 10000, "argument": 17, "bad_args": [17], "normal_args": [18]}
        return make(entity, body, facts, prefix)
    expressions = ("(x+y)*3", "x*x+y", "x*y+x", "(x+1)*(y+2)", "x*(y+1)+y", "x+x+y", "(x+y)+(x*y)", "x*4-y",
                   "(x-y)+x", "x-y+y*x", "x*x-y", "x*(x+1)+y", "(x+y)*(x+y)", "(x+y)*2+x", "x*2+y*3", "(x-y)*(x+y)")
    expression = expressions[index - 16]
    width = (16, 24, 32)[index % 3]
    typ, maximum = f"uint{width}", (1 << width) - 1
    which = index % 4
    extra, setup, calculation = "", "", f"return {expression};"
    if which == 0:
        setup = f"{typ}[2] memory a=[x,y];"
        calculation = "return " + expression.replace("x", "a[0]").replace("y", "a[1]") + ";"
    elif which == 1:
        extra = f"struct Args{{{typ} left;{typ} right;}}"
        setup = "Args memory a=Args(x,y);"
        calculation = "return " + expression.replace("x", "a.left").replace("y", "a.right") + ";"
    elif which == 2:
        inner = f"return {expression};"
        if vulnerable:
            inner = f"unchecked{{{inner}}}"
        extra = f"function helper{entity}({typ} x,{typ} y)private pure returns({typ}){{{inner}}}"
        calculation = f"return helper{entity}(x,y);"
    else:
        setup = f"{typ} z=x;"
        calculation = f"for({typ} i=0;i<1;i++){{z=" + expression.replace("x", "z") + ";}return z;"
    if vulnerable and which != 2:
        calculation = f"unchecked{{{calculation}}}"
    body = extra + f"function {f}({typ} x,{typ} y)external pure returns({typ}){{{setup}{calculation}}}"
    wrapped = eval(expression, {"__builtins__": {}}, {"x": maximum, "y": 2}) % (1 << width)
    normal = eval(expression, {"__builtins__": {}}, {"x": 6, "y": 2})
    facts = {"mechanism": "arithmetic", "function": f, "signature": f + f"({typ},{typ})", "width": width,
             "bad_args": [maximum, 2], "good_args": [6, 2], "wrapped_result": wrapped, "normal_result": normal}
    return make(entity, body, facts)
