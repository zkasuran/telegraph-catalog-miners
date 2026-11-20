# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Original training-only pathways verified before dataset integration.

No validation or audit input is accepted. Variants and vulnerable/fixed siblings
share their implementation group. Sources are AI-assisted, not human reviewed.
"""
import hashlib


def candidates():
    rows = []
    for design in range(8):
        for variant in range(2):
            entity = 70000 + design * 100 + variant
            f, c, rd = "f" + str(entity), "C" + str(entity), "read" + str(entity)
            owner, attacker, arg = 2000 + entity, 12000 + entity, 21 + 2 * variant
            for vulnerable in (True, False):
                if design < 4:
                    guard = "" if vulnerable else f"require(msg.sender==address({owner}));"
                    if design == 0:
                        code = f"uint256 value; function {f}(uint256 x) external {{if(x%2==0){{require(msg.sender==address({owner}));value=x;}}else{{{guard}value=x;}}}} function {rd}() external view returns(uint256){{return value;}}"
                    elif design == 1:
                        condition = f"msg.sender==address({owner})" + (f" || x=={arg}" if vulnerable else "")
                        code = f"struct Entry{{uint256 amount;uint256 revision;}} Entry entry; function {f}(uint256 x) external{{require({condition});entry.amount=x;entry.revision+=1;}}function {rd}()external view returns(uint256){{return entry.amount;}}"
                    elif design == 2:
                        code = f"mapping(uint256=>uint256) ledger;modifier nonzero(){{require(msg.sender!=address(0));_;}}modifier permission(uint256 x){{if(x%2==0){{require(msg.sender==address({owner}));}}else{{{guard}}}_;}}function {f}(uint256 x)external nonzero permission(x){{ledger[9]=x;}}function {rd}()external view returns(uint256){{return ledger[9];}}"
                    else:
                        code = f"uint256 value;function {f}(uint256 x)external{{write(x,msg.sender);}}function write(uint256 x,address actor)private{{" + ("require(actor!=address(0));" if vulnerable else f"require(actor==address({owner}));") + f"value=x;}}function {rd}()external view returns(uint256){{return value;}}"
                    facts = {"mechanism": "access_control", "function": f, "read_function": rd, "signature": f + "(uint256)", "owner": owner, "attacker": attacker, "argument": arg, "contract": c}
                else:
                    width = {4: 16, 5: 32, 6: 24, 7: 16}[design]
                    typ = "uint" + str(width)
                    maximum = (1 << width) - 1
                    expression = {4: "operands.left*operands.right", 5: "values[0]-values[1]", 6: "(x+y)*2", 7: "x*2+y"}[design]
                    declaration, setup = "", ""
                    if design == 4:
                        declaration = f"struct Operands{{{typ} left;{typ} right;}}"
                        setup = "Operands memory operands=Operands(x,y);"
                        bad, good, wrapped, normal = [maximum, 2], [2, 3], maximum - 1, 6
                    elif design == 5:
                        setup = f"{typ}[2] memory values=[x,y];"
                        bad, good, wrapped, normal = [0, 1], [4, 2], maximum, 2
                    elif design == 6:
                        bad, good, wrapped, normal = [maximum, 1], [2, 3], 0, 10
                    else:
                        bad, good, wrapped, normal = [maximum, 1], [2, 3], maximum, 7
                    body = f"unchecked{{return {expression};}}" if vulnerable else f"return {expression};"
                    code = declaration + f"function {f}({typ} x,{typ} y)external pure returns({typ}){{{setup}{body}}}"
                    facts = {"mechanism": "arithmetic", "function": f, "signature": f + f"({typ},{typ})", "width": width, "bad_args": bad, "good_args": good, "wrapped_result": wrapped, "normal_result": normal, "contract": c}
                source = "// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\npragma solidity 0.8.30;\n" + f"contract {c}{{{code}}}\n"
                rows.append({"id": f"new-train-{design}-{variant}-{int(vulnerable)}", "split": "training_candidate_only",
                             "proposed_group": f"new-training-design-{design}", "sibling_group": str(entity), "source": source,
                             "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "vulnerable": vulnerable, "facts": facts,
                             "authorship": "Original AI-assisted source candidates; no human review or accepted training integration claimed"})
    return rows
