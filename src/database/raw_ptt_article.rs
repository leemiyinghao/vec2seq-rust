use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct RawArticle {
    pub id: String,
    pub title: String,
    pub content: String,
    pub board: String,
    pub pushes: String,
    pub used: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Push {
    pub push_tag: String,
    pub push_userid: String,
    pub push_content: String,
    pub push_ipdatetime: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ArticleOrigin {
    PttOrigin { title: String, board: String },
    String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Article {
    pub content: String,
    pub pushes: Vec<String>,
    pub origin: ArticleOrigin,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ContentReply {
    pub content: String,
    pub replies: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedReplies {
    replies: Vec<u8>
}
use libflate::zlib::{Decoder, Encoder};
use std::io::{self, Read};
impl CompressedReplies{
    pub fn new() -> Self{
        CompressedReplies{
            replies: Vec::new(),
        }
    }
    pub fn new_from(replies: &Vec<String>) -> Self{
        CompressedReplies{
            replies: CompressedReplies::compress(replies),
        }
    }
    pub fn get(&self)->Vec<String>{
        CompressedReplies::decompress(&self.replies)
    }
    pub fn push(&mut self, reply: String){
        let mut replies = self.get();
        replies.push(reply);
        self.replies = CompressedReplies::compress(&replies);
    }
    fn compress(replies: &Vec<String>) -> Vec<u8>{
        let bytes = bincode::serialize(&replies).unwrap();
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        io::copy(&mut &bytes[..], &mut encoder).unwrap();
        encoder.finish().into_result().unwrap()
    }
    fn decompress(bytes: &Vec<u8>) -> Vec<String>{
        let mut decoder = Decoder::new(&bytes[..]).unwrap();
        let mut decoded_data = Vec::new();
        decoder.read_to_end(&mut decoded_data).unwrap();
        bincode::deserialize(&decoded_data).unwrap()
    }
}
