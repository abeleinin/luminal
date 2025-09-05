use crate::ast::{Attr, Operation};
use crate::parser::parse_output_shape_from_op;

use anyhow::{bail, Result};
use std::collections::HashMap;

use luminal::prelude::*;

type LowerFn = fn(&Operation, &mut Graph, &mut HashMap<String, GraphTensor>) -> Result<()>;

pub fn lower_op(
    op: &Operation,
    g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    match lookup(op.name.as_str()) {
        Some(f) => f(op, g, env),
        None => bail!("unsupported op {}", op.name),
    }
}

fn lookup(op: &str) -> Option<LowerFn> {
    let f = match op {
        // Unary
        "stablehlo.abs" => lower_unary_abs,
        "stablehlo.negate" => lower_unary_negate,
        "stablehlo.sqrt" => lower_unary_sqrt,
        "stablehlo.log" => lower_unary_log,
        "stablehlo.exponential" => lower_unary_exp,
        // Binary
        "stablehlo.add" => lower_bin_add,
        "stablehlo.subtract" => lower_bin_sub,
        "stablehlo.multiply" => lower_bin_mul,
        "stablehlo.divide" => lower_bin_div,
        "stablehlo.remainder" => lower_bin_rem,
        "stablehlo.maximum" => lower_bin_max,
        "stablehlo.minimum" => lower_bin_min,
        // Movement
        "stablehlo.reshape" => lower_reshape,
        "stablehlo.broadcast_in_dim" => lower_broadcast_in_dim,
        "stablehlo.concatenate" => lower_concatenate,
        // Constant
        "stablehlo.constant" => lower_constant,
        // Reduce
        "stablehlo.reduce" => lower_reduce,
        // Pseudo
        "return" => lower_return,
        _ => return None,
    };
    Some(f)
}

fn binary_with_numpy_broadcast<F>(mut a: GraphTensor, mut b: GraphTensor, f: F) -> GraphTensor
where
    F: Fn(GraphTensor, GraphTensor) -> GraphTensor,
{
    if a.shape.dims().is_empty() && !b.shape.dims().is_empty() {
        for &dim in b.shape.dims().iter() {
            a = a.expand_dim(0, dim);
        }
    } else if b.shape.dims().is_empty() && !a.shape.dims().is_empty() {
        for &dim in a.shape.dims().iter() {
            b = b.expand_dim(0, dim);
        }
    }
    f(a, b)
}

// Unary
fn lower_unary_abs(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.abs());
    Ok(())
}
fn lower_unary_negate(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), -x);
    Ok(())
}
fn lower_unary_sqrt(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.sqrt());
    Ok(())
}
fn lower_unary_log(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.log());
    Ok(())
}
fn lower_unary_exp(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.exp());
    Ok(())
}

// Binary
fn lower_bin_add(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l + r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_sub(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l - r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_mul(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l * r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_div(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l / r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_rem(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l % r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_max(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l.maximum(r));
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_min(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l.minimum(r));
    env.insert(op.result_name.clone(), y);
    Ok(())
}

// Movement
fn lower_reshape(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    let shape = parse_output_shape_from_op(&op.result_type_src);
    env.insert(op.result_name.clone(), x.reshape(shape));
    Ok(())
}

fn lower_broadcast_in_dim(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    let dims = match op.attributes.get("dims") {
        Some(Attr::IntVec(v)) => v.clone(),
        _ => bail!("broadcast_in_dim missing 'dims' attribute"),
    };
    let y = x.expand(dims);
    env.insert(op.result_name.clone(), y);
    Ok(())
}

fn lower_concatenate(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let dim = match op.attributes.get("dim") {
        Some(Attr::Int(i)) => *i as usize,
        _ => 0usize,
    };
    let y = a.concat_along(b, dim);
    env.insert(op.result_name.clone(), y);
    Ok(())
}

// Constant
fn lower_constant(
    op: &Operation,
    g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let v = match op.attributes.get("dense") {
        Some(Attr::Float(f)) => *f as f32,
        Some(Attr::Int(i)) => *i as f32,
        _ => bail!("constant missing 'dense' literal"),
    };
    let t = g.constant(v).retrieve();
    env.insert(op.result_name.clone(), t);
    Ok(())
}

// Reduce (sum only)
fn lower_reduce(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    match op.attributes.get("apply") {
        Some(Attr::Id(s)) if s == "stablehlo.add" => {
            let dims = match (op.attributes.get("dimensions"), op.attributes.get("dims")) {
                (Some(Attr::IntVec(v)), _) | (_, Some(Attr::IntVec(v))) => v.clone(),
                _ => vec![],
            };
            let y = x.sum(dims);
            env.insert(op.result_name.clone(), y);
            Ok(())
        }
        other => bail!("unsupported reduce.apply: {:?}", other),
    }
}

// return
fn lower_return(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    if let Some(src) = op.operands.get(0) {
        let y = env[src].retrieve();
        env.insert(src.clone(), y);
    }
    Ok(())
}
