use crate::ast::{Attr, AttrMap, Operation};
use crate::lexer::Tok;

use anyhow::{anyhow, bail, Result};
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
                    if s == "dim"
                        || s == "dims"
                        || s == "dim_numbers"
                        || s == "apply"
                        || s == "dense" =>
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

        if name == "stablehlo.convolution" {
            self.parse_convolution_attrs(&mut attrs)?;
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

    fn parse_convolution_attrs(&self, attrs: &mut AttrMap) -> Result<()> {
        // parse dim_numbers = [b, f, ...]x[o, i, ...]->[b, f, ...]
        if let Some(idx) = self.src.find("dim_numbers") {
            if let Some(start) = self.src[idx..].find('[') {
                let start = idx + start;
                let (a, p1) = extract_bracket_list(self.src, start)?;
                let after_a = &self.src[p1..];
                let x_pos = after_a
                    .find('x')
                    .ok_or_else(|| anyhow!("dim_numbers: missing 'x'"))?
                    + p1;
                let (b, p2) = extract_bracket_list(self.src, next_bracket_after(self.src, x_pos)?)?;
                let arrow_pos = self.src[p2..]
                    .find("->")
                    .ok_or_else(|| anyhow!("dim_numbers: missing '->'"))?
                    + p2;
                let (c, _p3) =
                    extract_bracket_list(self.src, next_bracket_after(self.src, arrow_pos)?)?;
                attrs.insert(
                    "dim_numbers".into(),
                    Attr::DimNumbers {
                        input: split_tags(a),
                        kernel: split_tags(b),
                        output: split_tags(c),
                    },
                );
            }
        }

        // parse window
        if let Some(wi) = self.src.find("window") {
            if let Some(open) = self.src[wi..].find('{') {
                let open = wi + open;
                if let Some(close_rel) = find_matching_brace(self.src, open) {
                    let body = &self.src[open + 1..close_rel];
                    if let Some(pi) = body.find("pad") {
                        if let Some(lb) = body[pi..].find('[') {
                            let pad_start = pi + lb;
                            let pads = parse_pad_pairs(&body[pad_start..])?;
                            attrs.insert("window_pad".into(), Attr::PadPairs(pads));
                        }
                    }
                    if let Some(si) = body.find("stride") {
                        if let Some(lb) = body[si..].find('[') {
                            let (vec, _) = parse_bracket_intvec(&body[si + lb..])?;
                            attrs.insert("stride".into(), Attr::IntVec(vec));
                        }
                    }
                    if let Some(bd) = body.find("base_dilations") {
                        if let Some(lb) = body[bd..].find('[') {
                            let (vec, _) = parse_bracket_intvec(&body[bd + lb..])?;
                            attrs.insert("base_dilations".into(), Attr::IntVec(vec));
                        }
                    }
                    if let Some(wd) = body.find("window_dilations") {
                        if let Some(lb) = body[wd..].find('[') {
                            let (vec, _) = parse_bracket_intvec(&body[wd + lb..])?;
                            attrs.insert("window_dilations".into(), Attr::IntVec(vec));
                        }
                    }
                }
            }
        }

        // group counts
        if let Some(bi) = self.src.find("batch_group_count") {
            if let Some((v, _)) = parse_trailing_int(&self.src[bi..]) {
                attrs.insert("batch_group_count".into(), Attr::Int(v as i64));
            }
        }
        if let Some(fi) = self.src.find("feature_group_count") {
            if let Some((v, _)) = parse_trailing_int(&self.src[fi..]) {
                attrs.insert("feature_group_count".into(), Attr::Int(v as i64));
            }
        }

        Ok(())
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
                // TODO: Use named_tensor instead of tensor
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

fn extract_bracket_list(s: &str, start: usize) -> Result<(String, usize)> {
    let mut depth = 0usize;
    let mut i = start;
    let bytes = s.as_bytes();
    while i < s.len() {
        let c = bytes[i] as char;
        if c == '[' {
            depth += 1;
            if depth == 1 {
                i += 1;
                let j = i;
                while i < s.len() {
                    let c2 = bytes[i] as char;
                    if c2 == ']' {
                        depth -= 1;
                        if depth == 0 {
                            let body = &s[j..i];
                            return Ok((body.trim().to_string(), i + 1));
                        }
                    }
                    i += 1;
                }
                break;
            }
        }
        i += 1;
    }
    Err(anyhow!("unclosed bracket list"))
}

fn next_bracket_after(s: &str, from: usize) -> Result<usize> {
    let rest = &s[from..];
    let off = rest.find('[').ok_or_else(|| anyhow!("expected '['"))?;
    Ok(from + off)
}

fn find_matching_brace(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    let b = s.as_bytes();
    for i in open..s.len() {
        let c = b[i] as char;
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

fn split_tags(s: String) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn parse_bracket_intvec(s: &str) -> Result<(Vec<usize>, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    let b = s.as_bytes();
    assert_eq!(b[0] as char, '[');
    i += 1;
    let mut num = String::new();
    while i < s.len() {
        let c = b[i] as char;
        match c {
            '0'..='9' => num.push(c),
            ',' => {
                if !num.is_empty() {
                    out.push(num.parse::<usize>().unwrap());
                    num.clear();
                }
            }
            ']' => {
                if !num.is_empty() {
                    out.push(num.parse::<usize>().unwrap());
                }
                return Ok((out, i + 1));
            }
            ' ' => {}
            _ => break,
        }
        i += 1;
    }
    Err(anyhow!("bad intvec"))
}

fn parse_pad_pairs(s: &str) -> Result<Vec<(usize, usize)>> {
    let mut pads = Vec::new();
    let mut i = 0;
    let b = s.as_bytes();
    while i < s.len() {
        if b[i] as char == '[' {
            i += 1;
            let mut a = String::new();
            let mut bnum = String::new();
            let mut reading_b = false;
            while i < s.len() {
                let c = b[i] as char;
                match c {
                    '0'..='9' => {
                        if !reading_b {
                            a.push(c)
                        } else {
                            bnum.push(c)
                        }
                    }
                    ',' => {
                        reading_b = true;
                    }
                    ']' => {
                        let lo = a.parse::<usize>().unwrap_or(0);
                        let hi = bnum.parse::<usize>().unwrap_or(0);
                        pads.push((lo, hi));
                        break;
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        if b[i] as char == ']' {
            break;
        }
        i += 1;
    }
    Ok(pads)
}

fn parse_trailing_int(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() && !bytes[i].is_ascii_digit() {
        i += 1;
    }

    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }

    if start < i {
        let v = s[start..i].parse::<usize>().ok()?;
        return Some((v, i));
    }

    None
}
