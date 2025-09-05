mod ast;
mod lexer;
mod lower;
mod parser;

use std::collections::HashMap;

use luminal::prelude::*;

use crate::{
    ast::Operation,
    lexer::Lexer,
    lower::lower_op,
    parser::{parse_func_args_line, Parser},
};

pub fn import_hlo(path: &str) -> (Box<Graph>, HashMap<String, GraphTensor>) {
    let contents = std::fs::read_to_string(path).expect("Failed to read file.");

    let mut cx = Box::new(Graph::new());
    let mut env: HashMap<String, GraphTensor> = HashMap::new();

    for line in contents.lines().map(str::trim) {
        if line.starts_with("func.func") {
            parse_func_args_line(line, &mut cx, &mut env);
            break;
        }
    }

    let mut ops: Vec<Operation> = Vec::new();
    for raw in contents.lines().map(str::trim) {
        if raw.starts_with('%') && raw.contains(" = stablehlo.") {
            let mut lx = Lexer::new(raw);
            let toks = lx.tokenize();
            let mut p = Parser::new(raw, toks);
            match p.parse_operation() {
                Ok(op) => ops.push(op),
                Err(e) => panic!("Parse error on op line:\n{}\n{:?}", raw, e),
            }
        } else if raw.starts_with("return") {
            let mut lx = Lexer::new(raw);
            let toks = lx.tokenize();
            let mut p = Parser::new(raw, toks);
            match p.parse_return() {
                Ok(ret) => ops.push(ret),
                Err(e) => panic!("Parse error on return line:\n{}\n{:?}", raw, e),
            }
        }
    }

    for op in ops.into_iter() {
        if let Err(e) = lower_op(&op, &mut cx, &mut env) {
            panic!("Lowering error for op {:?}: {}", op.name, e);
        }
    }

    (cx, env)
}
