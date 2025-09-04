use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use luminal::prelude::*;

// =============================
// Public entrypoint
// =============================

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

    let registry = OpRegistry::default();
    for op in ops.into_iter() {
        if let Err(e) = registry.lower(&op, &mut cx, &mut env) {
            panic!("Lowering error for op {:?}: {}", op.name, e);
        }
    }

    (cx, env)
}

// =============================
// Parse function args
// =============================

fn parse_func_args_line(line: &str, cx: &mut Graph, env: &mut HashMap<String, GraphTensor>) {
    if let Some((start_idx, end_idx)) = line.find('(').zip(line.find(')')) {
        let args_str = &line[start_idx + 1..end_idx];
        for arg in args_str.split(',') {
            let arg_tokens: Vec<&str> = arg.trim().split(':').collect();
            if let [arg_name, tensor_shape_str] = arg_tokens.as_slice() {
                let arg_name = arg_name.trim();
                let tensor_shape_str = tensor_shape_str.trim();
                let tensor_shape = parse_tensor_shape(tensor_shape_str);
                let tensor = cx.tensor(tensor_shape);
                env.insert(arg_name.to_string(), tensor);
            }
        }
    }
}

// =============================
// Tiny lexer
// =============================

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    PercentIdent(String),
    Ident(String),
    Integer(i64),
    Float(f64),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Eq,
    Less,
    Greater,
    Arrow,
}

struct Lexer<'a> {
    s: &'a str,
    i: usize,
    bytes: &'a [u8],
}

impl<'a> Lexer<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, i: 0, bytes: s.as_bytes() }
    }

    fn peek(&self) -> Option<u8> { self.bytes.get(self.i).copied() }
    fn bump(&mut self) -> Option<u8> { let b = self.peek()?; self.i += 1; Some(b) }
    fn eat_while<F: Fn(u8)->bool>(&mut self, f: F) -> &'a str {
        let start = self.i;
        while let Some(c) = self.peek() { if f(c) { self.i += 1; } else { break; } }
        &self.s[start..self.i]
    }

    fn skip_ws(&mut self) { self.eat_while(|c| c.is_ascii_whitespace()); }

    fn tokenize(&mut self) -> Vec<Tok> {
        let mut out = Vec::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() { self.skip_ws(); continue; }
            match c as char {
                '(' => { self.bump(); out.push(Tok::LParen); }
                ')' => { self.bump(); out.push(Tok::RParen); }
                '[' => { self.bump(); out.push(Tok::LBracket); }
                ']' => { self.bump(); out.push(Tok::RBracket); }
                '{' => { self.bump(); out.push(Tok::LBrace); }
                '}' => { self.bump(); out.push(Tok::RBrace); }
                ',' => { self.bump(); out.push(Tok::Comma); }
                ':' => { self.bump(); out.push(Tok::Colon); }
                '=' => { self.bump(); out.push(Tok::Eq); }
                '<' => { self.bump(); out.push(Tok::Less); }
                '>' => { self.bump(); out.push(Tok::Greater); }
                '-' => {
                    if self.i + 1 < self.bytes.len() && self.bytes[self.i + 1] == b'>' {
                        self.i += 2; out.push(Tok::Arrow);
                    } else {
                        let ident = self.lex_ident(); out.push(Tok::Ident(ident));
                    }
                }
                '%' => {
                    self.bump();
                    let body = self.eat_while(|c| matches!(c, b'a'..=b'z'|b'A'..=b'Z'|b'0'..=b'9'|b'_'|b'.'));
                    out.push(Tok::PercentIdent(format!("%{}", body)));
                }
                c if c.is_ascii_digit() => {
                    let num = self.lex_number(); out.push(num);
                }
                _ => {
                    let ident = self.lex_ident(); out.push(Tok::Ident(ident));
                }
            }
        }
        out
    }

    fn lex_ident(&mut self) -> String {
        let s = self.eat_while(|c| matches!(c, b'a'..=b'z'|b'A'..=b'Z'|b'0'..=b'9'|b'_'|b'.'|b'-'|b'@'|b'x'));
        s.to_string()
    }

    fn lex_number(&mut self) -> Tok {
        let start = self.i;
    
        // integer part
        self.eat_while(|c| c.is_ascii_digit());
    
        let mut is_float = false;
    
        // fractional part
        if self.peek() == Some(b'.') {
            is_float = true;
            self.bump();
            self.eat_while(|c| c.is_ascii_digit());
        }
    
        // exponent part
        if let Some(b'e') | Some(b'E') = self.peek() {
            is_float = true;
            self.bump();
    
            if let Some(b'+' | b'-') = self.peek() {
                self.bump();
            }
    
            self.eat_while(|c| c.is_ascii_digit());
        }
    
        let text = &self.s[start..self.i];
    
        if is_float {
            Tok::Float(text.parse::<f64>().unwrap())
        } else {
            Tok::Integer(text.parse::<i64>().unwrap())
        }
    }
}

// =============================
// AST + attributes
// =============================

#[derive(Clone, Debug)]
pub struct Operation {
    pub result_name: String,
    pub name: String,
    pub operands: Vec<String>,
    pub attributes: AttrMap,
    pub result_type_src: String,
}

#[derive(Clone, Debug)]
pub enum Attr {
    Int(i64),
    Float(f64),
    Id(String),
    IntList(Vec<i64>),
    IntVec(Vec<usize>),
    TensorSrc(String),
}

pub type AttrMap = HashMap<String, Attr>;

// =============================
// Parser (per-line for ops/return)
// =============================

struct Parser<'a> {
    src: &'a str,
    toks: Vec<Tok>,
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, toks: Vec<Tok>) -> Self { Self { src, toks, i: 0 } }

    fn at(&self, k: &Tok) -> bool { self.toks.get(self.i) == Some(k) }
    fn peek(&self) -> Option<&Tok> { self.toks.get(self.i) }
    fn bump(&mut self) -> Option<Tok> { if self.i < self.toks.len() { let t = self.toks[self.i].clone(); self.i += 1; Some(t) } else { None } }

    fn expect_percent_ident(&mut self) -> Result<String> {
        match self.bump() { Some(Tok::PercentIdent(s)) => Ok(s), _ => bail!("expected %ident") }
    }
    fn expect_ident(&mut self) -> Result<String> {
        match self.bump() { Some(Tok::Ident(s)) => Ok(s), _ => bail!("expected ident") }
    }
    fn expect(&mut self, want: Tok) -> Result<()> {
        match self.bump() { Some(t) if t == want => Ok(()), other => bail!("expected {:?}, got {:?}", want, other) }
    }

    fn parse_operation(&mut self) -> Result<Operation> {
        let result_name = self.expect_percent_ident()?;
        self.expect(Tok::Eq)?;
        let name = self.expect_ident()?;

        // Parse operand list
        let mut operands = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::PercentIdent(_)) => operands.push(self.expect_percent_ident()?),
                Some(Tok::Colon) => { self.bump(); break; }
                Some(Tok::Comma) | Some(Tok::LParen) => { self.bump(); }
                Some(Tok::Ident(s)) if s == "init" => { self.bump(); }
                Some(Tok::Ident(s)) if s == "dim" || s == "dims" || s == "apply" || s == "dense" => { break; }
                other => bail!("unexpected token in operand list: {:?}", other),
            }
        }

        let mut attrs: AttrMap = HashMap::new();

        while !matches!(self.peek(), None | Some(Tok::Arrow)) {
            match self.peek() {
                Some(Tok::Ident(s)) if s == "dims" => {
                    self.bump(); self.expect(Tok::Eq)?; let v = self.parse_intvec()?; attrs.insert("dims".into(), Attr::IntVec(v));
                }
                Some(Tok::Ident(s)) if s == "dim" => {
                    self.bump(); self.expect(Tok::Eq)?; let v = self.expect_integer()?; attrs.insert("dim".into(), Attr::Int(v));
                }
                Some(Tok::Ident(s)) if s == "applies" => {
                    self.bump(); let id = self.expect_ident()?;
                    attrs.insert("apply".into(), Attr::Id(id));
                    if let Some(Tok::Ident(s2)) = self.peek() { if s2 == "across" { self.bump(); } }
                    if let Some(Tok::Ident(s3)) = self.peek() { if s3 == "dimensions" { self.bump(); } }
                    if let Some(Tok::Eq) = self.peek() { self.bump(); let v = self.parse_intvec()?; attrs.insert("dimensions".into(), Attr::IntVec(v)); }
                }
                Some(Tok::Ident(s)) if s == "dense" => {
                    self.bump(); self.expect(Tok::Less)?;
                    if let Some(Tok::Float(v)) = self.peek() { attrs.insert("dense".into(), Attr::Float(v.clone())); }
                }
                _ => { self.bump(); }
            }
        }

        let mut result_type_src = String::new();
        if matches!(self.peek(), Some(Tok::Arrow)) {
            if let Some(pos) = self.src.find("->") { result_type_src = self.src[pos+2..].trim().to_string(); }
        }

        Ok(Operation { result_name, name, operands, attributes: attrs, result_type_src })
    }

    fn parse_return(&mut self) -> Result<Operation> {
        let name = String::from("return");
        let mut ret = String::new();
        while let Some(tok) = self.bump() {
            if let Tok::PercentIdent(s) = tok { ret = s; break; }
        }
        if ret.is_empty() { bail!("return missing %ident"); }
        Ok(Operation { result_name: "%_ret".into(), name, operands: vec![ret], attributes: HashMap::new(), result_type_src: String::new() })
    }

    fn parse_intvec(&mut self) -> Result<Vec<usize>> {
        self.expect(Tok::LBracket)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::Integer(i)) => { let v = *i as usize; self.bump(); out.push(v); },
                Some(Tok::Comma) => { self.bump(); },
                Some(Tok::RBracket) => { self.bump(); break; }
                other => bail!("bad intvec token: {:?}", other),
            }
        }
        Ok(out)
    }

    fn expect_integer(&mut self) -> Result<i64> {
        match self.bump() { Some(Tok::Integer(i)) => Ok(i), other => bail!("expected integer, got {:?}", other) }
    }
}

// =============================
// Type parsing (reuse your helpers)
// =============================

pub fn parse_tensor_shape(tensor_type_str: &str) -> Vec<usize> {
    if let Some(start) = tensor_type_str.find('<') {
        if let Some(end) = tensor_type_str.find('>') {
            let shape_str = &tensor_type_str[start + 1..end];

            if !shape_str.contains('x')
                && (shape_str.ends_with("f32")
                    || shape_str.ends_with("f16")
                    || shape_str.ends_with("i32")
                    || shape_str.ends_with("i64"))
            {
                return vec![1];
            }

            let dims: Vec<usize> = shape_str
                .split('x')
                .filter_map(|s| {
                    let s = s.trim();
                    if s.ends_with("f32")
                        || s.ends_with("f16")
                        || s.ends_with("i32")
                        || s.ends_with("i64")
                    {
                        None
                    } else {
                        s.parse::<usize>().ok()
                    }
                })
                .collect();

            if dims.is_empty() { vec![1] } else { dims }
        } else {
            panic!("Malformed tensor type: missing '>' in {}", tensor_type_str);
        }
    } else {
        panic!("Malformed tensor type: missing '<' in {}", tensor_type_str);
    }
}

pub fn parse_output_shape_from_op(op_line: &str) -> Vec<usize> {
    if let Some(tensor_start) = op_line.find("tensor<") {
        let tensor_end = op_line[tensor_start..]
            .find('>')
            .map(|pos| tensor_start + pos + 1)
            .unwrap_or(op_line.len());
        let tensor_type = &op_line[tensor_start..tensor_end];
        parse_tensor_shape(tensor_type)
    } else { panic!("No tensor type found after '->' in: {}", op_line); }
}

// =============================
// Lowering registry
// =============================

type LowerFn = fn(&Operation, &mut Graph, &mut HashMap<String, GraphTensor>) -> Result<()>;

#[derive(Default)]
struct OpRegistry { m: HashMap<&'static str, LowerFn> }

impl OpRegistry {
    fn default() -> Self {
        let mut this = Self { m: HashMap::new() };
        // Unary
        this.m.insert("stablehlo.abs", lower_unary_abs);
        this.m.insert("stablehlo.negate", lower_unary_negate);
        this.m.insert("stablehlo.sqrt", lower_unary_sqrt);
        this.m.insert("stablehlo.log", lower_unary_log);
        this.m.insert("stablehlo.exponential", lower_unary_exp);
        // Binary
        this.m.insert("stablehlo.add", lower_bin_add);
        this.m.insert("stablehlo.subtract", lower_bin_sub);
        this.m.insert("stablehlo.multiply", lower_bin_mul);
        this.m.insert("stablehlo.divide", lower_bin_div);
        this.m.insert("stablehlo.remainder", lower_bin_rem);
        this.m.insert("stablehlo.maximum", lower_bin_max);
        this.m.insert("stablehlo.minimum", lower_bin_min);
        // Movement
        this.m.insert("stablehlo.reshape", lower_reshape);
        this.m.insert("stablehlo.broadcast_in_dim", lower_broadcast_in_dim);
        this.m.insert("stablehlo.concatenate", lower_concatenate);
        // Constant
        this.m.insert("stablehlo.constant", lower_constant);
        // Reduce
        this.m.insert("stablehlo.reduce", lower_reduce);
        // Pseudo op for return
        this.m.insert("return", lower_return);
        this
    }

    fn lower(&self, op: &Operation, g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
        match self.m.get(op.name.as_str()) {
            Some(f) => f(op, g, env),
            None => bail!("unsupported op {}", op.name),
        }
    }
}

// =============================
// Lowering helpers
// =============================

fn get(env: &HashMap<String, GraphTensor>, name: &str) -> Result<GraphTensor> {
    env.get(name).cloned().ok_or_else(|| anyhow!("unknown value {}", name))
}

fn binary_with_numpy_broadcast<F>(mut a: GraphTensor, mut b: GraphTensor, f: F) -> GraphTensor
where
    F: Fn(GraphTensor, GraphTensor) -> GraphTensor,
{
    if a.shape.dims().is_empty() && !b.shape.dims().is_empty() {
        for &dim in b.shape.dims().iter() { a = a.expand_dim(0, dim); }
    } else if b.shape.dims().is_empty() && !a.shape.dims().is_empty() {
        for &dim in a.shape.dims().iter() { b = b.expand_dim(0, dim); }
    }
    f(a, b)
}

// =============================
// Lowering implementations
// =============================

// Unary
fn lower_unary_abs(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?; env.insert(op.result_name.clone(), x.abs()); Ok(())
}
fn lower_unary_negate(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?; env.insert(op.result_name.clone(), -x); Ok(())
}
fn lower_unary_sqrt(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?; env.insert(op.result_name.clone(), x.sqrt()); Ok(())
}
fn lower_unary_log(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?; env.insert(op.result_name.clone(), x.log()); Ok(())
}
fn lower_unary_exp(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?; env.insert(op.result_name.clone(), x.exp()); Ok(())
}

// Binary
fn lower_bin_add(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let y = binary_with_numpy_broadcast(a, b, |l, r| l + r); env.insert(op.result_name.clone(), y); Ok(())
}
fn lower_bin_sub(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let y = binary_with_numpy_broadcast(a, b, |l, r| l - r); env.insert(op.result_name.clone(), y); Ok(())
}
fn lower_bin_mul(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let y = binary_with_numpy_broadcast(a, b, |l, r| l * r); env.insert(op.result_name.clone(), y); Ok(())
}
fn lower_bin_div(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let y = binary_with_numpy_broadcast(a, b, |l, r| l / r); env.insert(op.result_name.clone(), y); Ok(())
}
fn lower_bin_rem(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let y = binary_with_numpy_broadcast(a, b, |l, r| l % r); env.insert(op.result_name.clone(), y); Ok(())
}
fn lower_bin_max(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let y = binary_with_numpy_broadcast(a, b, |l, r| l.maximum(r)); env.insert(op.result_name.clone(), y); Ok(())
}
fn lower_bin_min(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let y = binary_with_numpy_broadcast(a, b, |l, r| l.minimum(r)); env.insert(op.result_name.clone(), y); Ok(())
}

// Movement
fn lower_reshape(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?;
    let shape = parse_output_shape_from_op(&op.result_type_src);
    env.insert(op.result_name.clone(), x.reshape(shape)); Ok(())
}

fn lower_broadcast_in_dim(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?;
    let dims = match op.attributes.get("dims") { Some(Attr::IntVec(v)) => v.clone(), _ => bail!("broadcast_in_dim missing 'dims' attribute") };
    let y = x.expand(dims);
    env.insert(op.result_name.clone(), y);
    Ok(())
}

fn lower_concatenate(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let a = get(env, &op.operands[0])?; let b = get(env, &op.operands[1])?;
    let dim = match op.attributes.get("dim") { Some(Attr::Int(i)) => *i as usize, _ => 0usize };
    let y = a.concat_along(b, dim);
    env.insert(op.result_name.clone(), y);
    Ok(())
}

// Constant
fn lower_constant(op: &Operation, g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
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
fn lower_reduce(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let x = get(env, &op.operands[0])?;
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
fn lower_return(op: &Operation, _g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    if let Some(src) = op.operands.get(0) {
        let y = get(env, src)?.retrieve();
        env.insert(src.clone(), y);
    }
    Ok(())
}
