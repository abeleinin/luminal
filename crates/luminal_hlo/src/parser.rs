use crate::ast::{Attr, AttrMap, Operation};
use crate::lexer::Tok;

use anyhow::{bail, Result};
use std::collections::HashMap;

use luminal::prelude::*;

pub struct Parser<'a> {
    src: &'a str,
    toks: Vec<Tok>,
    i: usize,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a str, toks: Vec<Tok>) -> Self {
        Self { src, toks, i: 0 }
    }

    pub fn parse_operation(&mut self) -> Result<Operation> {
        let result_name = self.expect_percent_ident()?;
        self.expect(Tok::Eq)?;
        let name = self.expect_ident()?;

        // Parse operand list
        let mut operands = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::PercentIdent(_)) => operands.push(self.expect_percent_ident()?),
                Some(Tok::Colon) => {
                    self.bump();
                    break;
                }
                Some(Tok::Comma) | Some(Tok::LParen) | Some(Tok::RParen) => {
                    self.bump();
                }
                Some(Tok::Ident(s)) if s == "init" => {
                    self.bump();
                }
                Some(Tok::Ident(s))
                    if s == "dim" || s == "dims" || s == "apply" || s == "dense" =>
                {
                    break;
                }
                other => bail!("unexpected token in operand list: {:?}", other),
            }
        }

        let mut attrs: AttrMap = HashMap::new();

        while !matches!(self.peek(), None | Some(Tok::Arrow)) {
            match self.peek() {
                Some(Tok::Ident(s)) if s == "dims" => {
                    self.bump();
                    self.expect(Tok::Eq)?;
                    let v = self.parse_intvec()?;
                    attrs.insert("dims".into(), Attr::IntVec(v));
                }
                Some(Tok::Ident(s)) if s == "dim" => {
                    self.bump();
                    self.expect(Tok::Eq)?;
                    let v = self.expect_integer()?;
                    attrs.insert("dim".into(), Attr::Int(v));
                }
                Some(Tok::Ident(s)) if s == "applies" => {
                    self.bump();
                    let id = self.expect_ident()?;
                    attrs.insert("apply".into(), Attr::Id(id));
                    if let Some(Tok::Ident(s2)) = self.peek() {
                        if s2 == "across" {
                            self.bump();
                        }
                    }
                    if let Some(Tok::Ident(s3)) = self.peek() {
                        if s3 == "dimensions" {
                            self.bump();
                        }
                    }
                    if let Some(Tok::Eq) = self.peek() {
                        self.bump();
                        let v = self.parse_intvec()?;
                        attrs.insert("dimensions".into(), Attr::IntVec(v));
                    }
                }
                Some(Tok::Ident(s)) if s == "dense" => {
                    self.bump();
                    self.expect(Tok::Less)?;
                    if let Some(Tok::Float(v)) = self.peek() {
                        attrs.insert("dense".into(), Attr::Float(v.clone()));
                    }
                }
                _ => {
                    self.bump();
                }
            }
        }

        let mut result_type_src = String::new();
        if matches!(self.peek(), Some(Tok::Arrow)) {
            if let Some(pos) = self.src.find("->") {
                result_type_src = self.src[pos + 2..].trim().to_string();
            }
        }

        Ok(Operation {
            result_name,
            name,
            operands,
            attributes: attrs,
            result_type_src,
        })
    }

    pub fn parse_return(&mut self) -> Result<Operation> {
        let name = String::from("return");
        let mut ret = String::new();
        while let Some(tok) = self.bump() {
            if let Tok::PercentIdent(s) = tok {
                ret = s;
                break;
            }
        }
        if ret.is_empty() {
            bail!("return missing %ident");
        }
        Ok(Operation {
            result_name: "%_ret".into(),
            name,
            operands: vec![ret],
            attributes: HashMap::new(),
            result_type_src: String::new(),
        })
    }

    fn parse_intvec(&mut self) -> Result<Vec<usize>> {
        self.expect(Tok::LBracket)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::Integer(i)) => {
                    let v = *i as usize;
                    self.bump();
                    out.push(v);
                }
                Some(Tok::Comma) => {
                    self.bump();
                }
                Some(Tok::RBracket) => {
                    self.bump();
                    break;
                }
                other => bail!("bad intvec token: {:?}", other),
            }
        }
        Ok(out)
    }

    fn expect_integer(&mut self) -> Result<i64> {
        match self.bump() {
            Some(Tok::Integer(i)) => Ok(i),
            other => bail!("expected integer, got {:?}", other),
        }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i)
    }

    fn bump(&mut self) -> Option<Tok> {
        if self.i < self.toks.len() {
            let t = self.toks[self.i].clone();
            self.i += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect_percent_ident(&mut self) -> Result<String> {
        match self.bump() {
            Some(Tok::PercentIdent(s)) => Ok(s),
            _ => bail!("expected %ident"),
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            _ => bail!("expected ident"),
        }
    }

    fn expect(&mut self, want: Tok) -> Result<()> {
        match self.bump() {
            Some(t) if t == want => Ok(()),
            other => bail!("expected {:?}, got {:?}", want, other),
        }
    }
}

pub fn parse_func_args_line(line: &str, cx: &mut Graph, env: &mut HashMap<String, GraphTensor>) {
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

pub fn parse_output_shape_from_op(op_line: &str) -> Vec<usize> {
    if let Some(tensor_start) = op_line.find("tensor<") {
        let tensor_end = op_line[tensor_start..]
            .find('>')
            .map(|pos| tensor_start + pos + 1)
            .unwrap_or(op_line.len());
        let tensor_type = &op_line[tensor_start..tensor_end];
        parse_tensor_shape(tensor_type)
    } else {
        panic!("No tensor type found after '->' in: {}", op_line);
    }
}

fn parse_tensor_shape(tensor_type_str: &str) -> Vec<usize> {
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

            if dims.is_empty() {
                vec![1]
            } else {
                dims
            }
        } else {
            panic!("Malformed tensor type: missing '>' in {}", tensor_type_str);
        }
    } else {
        panic!("Malformed tensor type: missing '<' in {}", tensor_type_str);
    }
}
