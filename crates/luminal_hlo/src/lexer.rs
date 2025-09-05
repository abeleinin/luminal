#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
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

pub struct Lexer<'a> {
    s: &'a str,
    i: usize,
    bytes: &'a [u8],
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str) -> Self {
        Self {
            s,
            i: 0,
            bytes: s.as_bytes(),
        }
    }

    pub fn tokenize(&mut self) -> Vec<Tok> {
        let mut out = Vec::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.skip_ws();
                continue;
            }
            match c as char {
                '(' => {
                    self.bump();
                    out.push(Tok::LParen);
                }
                ')' => {
                    self.bump();
                    out.push(Tok::RParen);
                }
                '[' => {
                    self.bump();
                    out.push(Tok::LBracket);
                }
                ']' => {
                    self.bump();
                    out.push(Tok::RBracket);
                }
                '{' => {
                    self.bump();
                    out.push(Tok::LBrace);
                }
                '}' => {
                    self.bump();
                    out.push(Tok::RBrace);
                }
                ',' => {
                    self.bump();
                    out.push(Tok::Comma);
                }
                ':' => {
                    self.bump();
                    out.push(Tok::Colon);
                }
                '=' => {
                    self.bump();
                    out.push(Tok::Eq);
                }
                '<' => {
                    self.bump();
                    out.push(Tok::Less);
                }
                '>' => {
                    self.bump();
                    out.push(Tok::Greater);
                }
                '-' => {
                    if self.i + 1 < self.bytes.len() && self.bytes[self.i + 1] == b'>' {
                        self.i += 2;
                        out.push(Tok::Arrow);
                    } else {
                        let ident = self.lex_ident();
                        out.push(Tok::Ident(ident));
                    }
                }
                '%' => {
                    self.bump();
                    let body = self
                        .eat_while(|c| matches!(c, b'a'..=b'z'|b'A'..=b'Z'|b'0'..=b'9'|b'_'|b'.'));
                    out.push(Tok::PercentIdent(format!("%{}", body)));
                }
                c if c.is_ascii_digit() => {
                    let num = self.lex_number();
                    out.push(num);
                }
                _ => {
                    let ident = self.lex_ident();
                    out.push(Tok::Ident(ident));
                }
            }
        }
        out
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.i).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.i += 1;
        Some(b)
    }

    fn eat_while<F: Fn(u8) -> bool>(&mut self, f: F) -> &'a str {
        let start = self.i;
        while let Some(c) = self.peek() {
            if f(c) {
                self.i += 1;
            } else {
                break;
            }
        }
        &self.s[start..self.i]
    }

    fn skip_ws(&mut self) {
        self.eat_while(|c| c.is_ascii_whitespace());
    }

    fn lex_ident(&mut self) -> String {
        let s = self
            .eat_while(|c| matches!(c, b'a'..=b'z'|b'A'..=b'Z'|b'0'..=b'9'|b'_'|b'.'|b'-'|b'@'));
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
