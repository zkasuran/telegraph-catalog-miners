# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fresh audit implementations. No exposed audit source is reused for fitting.

Each group names a distinct storage/authorization or arithmetic execution path.
The recipe and original train/validation data are frozen before these fixtures.
These are AI-assisted original source groups, not independent human reviewers.
"""


def access(index, entity, vulnerable):
    c, f, rd = (p + str(entity) for p in ("C", "f", "read"))
    owner, attacker, argument = 100 + entity, 10100 + entity, 17 + entity % 100
    check = "" if vulnerable else f"require(msg.sender == address({owner}));"
    declarations, store, getter = "uint256 private value;", "value=x;", "return value;"
    signature, args, normal_args = f + "(uint256)", [argument], [argument + 1]
    extra, prefix, suffix, head = "", "", "", ""
    purposes = [
        "mapping entry authorization", "structured field authorization", "fixed array modifier authorization",
        "typed authority helper", "address XOR authorization", "hashed caller authorization",
        "untrusted asserted authority", "authorization before batch storage updates", "versioned record authorization",
        "owner or explicit administrator", "inherited modifier authorization", "virtual authorization hook",
        "authorization in internal writer", "selector check is not caller authority", "chain identity is not caller authority",
        "authorization before assembly storage write",
    ]
    if index == 0:
        declarations, store, getter = "mapping(uint256=>uint256) private values;", "values[3]=x;", "return values[3];"
    elif index == 1:
        declarations = "struct Cell {uint256 value;} Cell private cell;"
        store, getter = "cell.value=x;", "return cell.value;"
        check = "" if vulnerable else f"if(msg.sender != address({owner})) revert();"
    elif index == 2:
        declarations = "uint256[2] private cells;" + f"modifier gate(){{{check}_;}}"
        check, suffix = "", " gate"
        store, getter = "cells[1]=x;", "return cells[1];"
    elif index == 3:
        declarations += f"address private constant OWNER=address({owner}); function permitted(address who) private pure returns(bool){{return who==OWNER;}}"
        check = "" if vulnerable else "require(permitted(msg.sender));"
    elif index == 4:
        check = "" if vulnerable else f"require((uint160(msg.sender)^uint160(address({owner})))==0);"
    elif index == 5:
        check = "" if vulnerable else f"require(keccak256(abi.encodePacked(msg.sender))==keccak256(abi.encodePacked(address({owner}))));"
    elif index == 6:
        signature, args, normal_args = f + "(uint256,uint160)", [argument, owner], [argument + 1, owner]
        head = ",uint160 asserted"
        check = f"require(asserted=={owner});" + ("" if vulnerable else "require(uint160(msg.sender)==asserted);")
    elif index == 7:
        declarations, store, getter = "uint256[2] private cells;", "for(uint256 i=0;i<2;i++){cells[i]=x;}", "return cells[0];"
    elif index == 8:
        declarations += "uint256 private revision;"
        store = "value=x;revision+=1;"
    elif index == 9:
        declarations = "uint128 private value;"
        check = "" if vulnerable else f"require(msg.sender==address({owner}) || msg.sender==address({owner+1}));"
        store = "value=uint128(x);"
    elif index == 10:
        prefix = f"contract Authority{entity} {{ modifier gate() {{{check}_;}} }}"
        extra, check, suffix = f" is Authority{entity}", "", " gate"
    elif index == 11:
        prefix = f"abstract contract Authority{entity} {{ function permitted() internal view virtual returns(bool); }}"
        extra = f" is Authority{entity}"
        declarations += "function permitted() internal view override returns(bool){return " + ("true" if vulnerable else f"msg.sender==address({owner})") + ";}"
        check = "require(permitted());"
        # Solidity flags the always-true implementation as unnecessarily view.
        if vulnerable:
            declarations = declarations.replace("internal view override", "internal pure override")
    elif index == 12:
        declarations += f"function commit(uint256 x) private {{{check}value=x;}}"
        check, store = "", "commit(x);"
    elif index == 13:
        check = f"require(msg.sig==this.{f}.selector);" + check
    elif index == 14:
        check = "require(block.chainid==1);" + check
    elif index == 15:
        declarations = "function commit(uint256 x) private {require(x!=0);" + check + "assembly{sstore(0,x)}}"
        check, store, getter = "", "commit(x);", "uint256 x;assembly{x:=sload(0)} return x;"
    source = prefix + f"contract {c}{extra} {{{declarations} function {f}(uint256 x{head}) external{suffix} {{{check}{store}}} function {rd}() external view returns(uint256) {{{getter}}}}}"
    return source, {"mechanism": "access_control", "function": f, "read_function": rd,
                    "owner": owner, "attacker": attacker, "argument": argument,
                    "bad_args": args, "normal_args": normal_args, "signature": signature,
                    "contract": c, "design_reason": purposes[index]}


def arithmetic(index, entity, vulnerable):
    c, f = "C" + str(entity), "f" + str(entity)
    # Explicit boundary oracles are arithmetic facts, not source-label annotations.
    specifications = [
        (16, "x-y", "x>=y", [0, 1], [4, 2], (1 << 16)-1, 2, "subtraction boundary"),
        (8, "x**y", "x<=1 || y<2 || (y==2 && x<=15) || (y==3 && x<=6)", [16, 2], [2, 3], 0, 8, "exponentiation boundary"),
        (8, "x+1", "x<255", [255], [2], 0, 3, "increment boundary"),
        (8, "x-1", "x>0", [0], [2], 255, 1, "decrement boundary"),
        (16, "x*y/z", "y==0 || x<=65535/y", [65535, 2, 2], [4, 2, 2], 32767, 4, "intermediate product before division"),
        (16, "(x+y)/2", "x<=65535-y", [65535, 65535], [2, 4], 32767, 3, "average intermediate overflow"),
        (16, "x*y+17", "y==0 || x<=(65535-17)/y", [65535, 2], [2, 3], 15, 23, "multiply-add compound bound"),
        (32, "x*1000", "x<=4294967", [(1 << 32)-1], [2], (1 << 32)-1000, 2000, "decimal scaling overflow"),
        (8, "uint8(x)", "x<=255", [256], [3], 0, 3, "wide-to-byte truncation"),
        (16, "uint16(x)", "x<=65535", [65536], [9], 0, 9, "wide-to-word truncation"),
        (8, "uint8(x)+uint8(y)", "x<=255 && y<=255 && x+y<=255", [256, 0], [2, 3], 0, 5, "narrowing before addition"),
        (8, "x<<1", "x<=127", [128], [3], 0, 6, "left-shift discarded high bit"),
        (16, "(x<<8)|y", "x<=255 && y<=255", [256, 1], [2, 3], 1, 515, "bitfield packing bound"),
        (16, "x*y*y", "y==0 || x<=65535/y/y", [32768, 2], [2, 3], 0, 18, "sequential product overflow"),
        (8, "x+y", "x<=255-y", [255, 2], [2, 3], 1, 5, "array accumulation boundary"),
        (32, "x-y-5", "x>=y && x-y>=5", [4, 0], [9, 2], (1 << 32)-1, 2, "fee subtraction underflow"),
    ]
    width, expression, predicate, bad, good, wrapped, normal, purpose = specifications[index]
    typ = f"uint{width}"
    parameter_type = "uint256" if index in (8, 9, 10) else typ
    params = ",".join(parameter_type + " " + v for v in "xyz"[:len(bad)])
    check = "" if vulnerable else "require(" + predicate + ");"
    if index == 14:
        body = check + "uint8[2] memory values=[x,y];uint8 total;for(uint256 i=0;i<2;i++){unchecked{total+=values[i];}}return total;"
    else:
        body = check + "unchecked{return " + expression + ";}"
    source = f"contract {c} {{function {f}({params}) external pure returns({typ}) {{{body}}}}}"
    return source, {"mechanism": "arithmetic", "function": f, "width": width,
                    "operation": purpose, "bad_args": bad, "good_args": good,
                    "wrapped_result": wrapped, "normal_result": normal,
                    "signature": f + "(" + ",".join([parameter_type] * len(bad)) + ")",
                    "contract": c, "design_reason": purpose}


def source_for(index, entity, vulnerable):
    source, facts = access(index, entity, vulnerable) if index < 16 else arithmetic(index - 16, entity, vulnerable)
    return "// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\npragma solidity 0.8.30;\n" + source + "\n", facts
