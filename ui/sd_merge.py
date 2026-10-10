"""Safe, preview-only merge of a generated Behave decorator into an existing SD."""
import ast
import copy
import re

class MergeError(ValueError):
    pass

def _normalize_pattern(pattern):
    # Compare parameterized expressions by literal text, not parameter variable names.
    return re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", "{PARAM}", pattern).strip().lower()

def _decorator_info(node):
    for d in node.decorator_list:
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in ('given','when','then') and d.args:
            if isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value,str):
                yield d.func.id, d.args[0].value

def merge_existing_sd(original_source, generated_source, *, requested_type='then'):
    """Return a full updated SD preview; never write to disk.

    Requires a matching parameterized pattern, a missing decorator, and exactly
    one generated implementation. Preserves the original source verbatim.
    """
    if requested_type not in ('given','when','then'):
        raise MergeError('Unsupported decorator.')
    try:
        original=ast.parse(original_source)
        generated=ast.parse(generated_source)
    except SyntaxError as e:
        raise MergeError(f'Invalid Python source: {e}') from e
    new_functions=[n for n in generated.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if len(new_functions)!=1 or isinstance(new_functions[0],ast.AsyncFunctionDef):
        raise MergeError('Expected exactly one synchronous generated step function.')
    new_fn=copy.deepcopy(new_functions[0])
    new_decorators=list(_decorator_info(new_fn))
    if len(new_decorators)!=1 or new_decorators[0][0]!=requested_type:
        raise MergeError('Generated function must register exactly the requested decorator.')
    new_pattern=new_decorators[0][1]
    if not re.search(r'\{[A-Za-z_][A-Za-z0-9_]*\}',new_pattern):
        raise MergeError('Expected a parameterized Behave keyword pattern.')
    matches=[]
    registered=[]
    for n in original.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for decorator,pattern in _decorator_info(n):
                registered.append((decorator,pattern))
                if _normalize_pattern(pattern)==_normalize_pattern(new_pattern):
                    matches.append((decorator,pattern,n))
    if not matches:
        raise MergeError('The existing SD does not contain the matching keyword pattern. Refusing to merge.')
    if any(d==requested_type for d,_,_ in matches):
        raise MergeError(f'@{requested_type} is already registered for this keyword. No merge needed.')
    # Merge decorator metadata without importing confidential runtime modules.
    # This updates only the literal decorator arguments on the matching original
    # function(s), not their bodies, imports, comments, or other SD definitions.
    old_fn=matches[0][2]
    def decorator_arg(fn, name):
        found=[d for d in fn.decorator_list if isinstance(d,ast.Call)
               and isinstance(d.func,ast.Name) and d.func.id==name]
        if len(found)!=1 or len(found[0].args)!=1 or found[0].keywords:
            raise MergeError(f'Expected one literal @{name} decorator.')
        try: value=ast.literal_eval(found[0].args[0])
        except (ValueError,TypeError,SyntaxError):
            raise MergeError(f'Cannot inspect @{name} metadata safely.')
        return found[0].args[0],value
    def platforms(value):
        if not isinstance(value,str):
            raise MergeError('Platform must be a comma-separated string.')
        vals=[v.strip() for v in value.split(',')]
        if not vals or any(not v or not re.fullmatch(r'[A-Za-z0-9_.-]+',v) for v in vals):
            raise MergeError('Invalid platform metadata.')
        return vals
    def tams(value):
        if not isinstance(value,list) or not value or any(type(v)!=int or v<0 for v in value):
            raise MergeError('TAMS metadata must be a nonempty list of nonnegative integer IDs.')
        return value
    def unique(items):
        return list(dict.fromkeys(items))
    old_pnode,old_p=decorator_arg(old_fn,'platform')
    old_tnode,old_t=decorator_arg(old_fn,'TAMS')
    _,new_p=decorator_arg(new_fn,'platform')
    _,new_t=decorator_arg(new_fn,'TAMS')
    merged_p=','.join(unique(platforms(old_p)+platforms(new_p)))
    merged_t=unique(tams(old_t)+tams(new_t))
    # Update the existing original decorator source by offsets, preserving
    # original implementation verbatim. Other same-pattern functions are rejected
    # if their metadata differs, to avoid ambiguous partial updates.
    for _,_,fn in matches:
        pnode,pvalue=decorator_arg(fn,'platform')
        tnode,tvalue=decorator_arg(fn,'TAMS')
        if pvalue!=old_p or tvalue!=old_t:
            raise MergeError('Matching original functions have inconsistent metadata.')
    # The source edits are applied after the generated function is prepared.
    # Reuse the existing pattern exactly, including the placeholder name.
    old_pattern=matches[0][1]
    old_names=re.findall(r'\{([A-Za-z_][A-Za-z0-9_]*)\}',old_pattern)
    new_names=re.findall(r'\{([A-Za-z_][A-Za-z0-9_]*)\}',new_pattern)
    if len(old_names)!=len(new_names) or len(set(new_names))!=len(new_names):
        raise MergeError('Incompatible Behave parameter signatures.')
    mapping=dict(zip(new_names,old_names))
    if len(set(old_names))!=len(old_names):
        raise MergeError('Ambiguous original parameter names.')
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in mapping:
                node.id=mapping[node.id]
            return node
        def visit_arg(self,node):
            if node.arg in mapping:
                node.arg=mapping[node.arg]
            return node
    new_fn=Rename().visit(new_fn)
    # Adjust generated registration to use the original pattern verbatim.
    for dec in new_fn.decorator_list:
        if isinstance(dec,ast.Call) and isinstance(dec.func,ast.Name) and dec.func.id==requested_type:
            dec.args[0]=ast.Constant(value=old_pattern)
    existing_names={n.name for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    proposed='verify_'+requested_type+'_step'
    suffix=2
    while proposed in existing_names:
        proposed=f'verify_{requested_type}_step_{suffix}';suffix+=1
    new_fn.name=proposed
    # Ensure required imports exist without rewriting the original source.
    existing_imports={alias.name for n in original.body if isinstance(n,ast.ImportFrom) and n.module=='behave' for alias in n.names}
    existing_managers={alias.name for n in original.body if isinstance(n,ast.ImportFrom) and n.module=='step_definitions_manager' for alias in n.names}
    additions=[]
    if requested_type not in existing_imports:
        additions.append(f'from behave import {requested_type}')
    needed=[x for x in ('TAMS','platform') if x not in existing_managers]
    if needed:
        additions.append('from step_definitions_manager import '+', '.join(needed))
    # Use merged metadata for the newly added function as well.
    for d in new_fn.decorator_list:
        if isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.args:
            if d.func.id=='platform': d.args[0]=ast.Constant(merged_p)
            if d.func.id=='TAMS': d.args[0]=ast.List(elts=[ast.Constant(x) for x in merged_t],ctx=ast.Load())
    # Replace only existing metadata argument spans (in reverse byte offset order).
    lines=original_source.splitlines(keepends=True)
    edits=[]
    for _,_,fn in matches:
        for name,newvalue in (('platform',repr(merged_p)),('TAMS',repr(merged_t))):
            node,_=decorator_arg(fn,name)
            if node.lineno!=node.end_lineno:
                raise MergeError('Multiline decorator metadata needs manual review.')
            edits.append((node.lineno-1,node.col_offset,node.end_col_offset,newvalue))
    for line,start,end,replacement in sorted(set(edits),reverse=True):
        original_line=lines[line]
        lines[line]=original_line[:start]+replacement+original_line[end:]
    updated_original=''.join(lines)
    new_code=ast.unparse(ast.fix_missing_locations(new_fn))
    combined=updated_original.rstrip()+'\n\n\n'+('\n'.join(additions)+'\n\n' if additions else '')+new_code+'\n'
    ast.parse(combined)
    return {'code':combined,'new_function':new_code,'target_pattern':old_pattern,
            'requested_decorator':requested_type,'function_name':proposed,
            'original_logic_preserved':True,'merged_platforms':merged_p,'merged_tams':merged_t}
