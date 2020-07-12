#[macro_use]
extern crate timeit;
use rand::Rng;
mod database;
mod math_tool;
mod sentence_embedding;
mod tasks;
mod tfidf;
use finalfusion::prelude::*;
use granne::{angular, BuildConfig, Builder, Granne, GranneBuilder, Index};
use indicatif::{HumanDuration, MultiProgress, ProgressBar, ProgressStyle};
use leveldb;
use leveldb::kv::KV;
use leveldb_sys;
use rusqlite::{params, Connection, Result};
use serde_json::Value;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::time::{Duration, Instant};

fn test() {
    timeit!({
        let mut result = math_tool::dot(
            &vec![1f32, 1f32, 1f32, 1f32, 1f32],
            &vec![1f32, 1f32, 1f32, 1f32, 1f32],
        )
        .unwrap();
        //println!("[1,1,1,1,1] dot [1,1,1,1,1] = {}", &result);
        assert_eq!(result, 5f32);
    });
    let mut rng = rand::thread_rng();
    timeit!({
        for i in 1..10000 {
            let mut vector: Vec<f32> = Vec::new();
            for j in 0..300 {
                let item: f32 = rng.gen();
                vector.push(if (item == 0f32) { 1f32 } else { item });
            }
            assert_ne!(vector[0], 0f32);
        }
    });
    timeit!({
        for i in 1..10000 {
            let mut vector: Vec<f32> = Vec::new();
            for j in 0..300 {
                let item: f32 = rng.gen();
                vector.push(if (item == 0f32) { 1f32 } else { item });
            }
            let mut result = math_tool::consine_similarity(&vector, &vector).unwrap();
            //println!("[0,0,0,0,0] - [1,1,1,1,1] = {}", result);
            assert_eq!(result, 1.0f32);
        }
    });
}
// fn test_sentence_sim() {
//     let mut tfidf_bufreader = BufReader::new(File::open("tfidf.bin").unwrap());
//     let mut buf: Vec<u8> = Vec::new();
//     tfidf_bufreader.read_to_end(&mut buf);
//     let tfidf = bincode::deserialize::<tfidf::TfIdf>(&buf).unwrap();
//     let mut ff_reader =
//         BufReader::new(File::open("../cut_corpus/finalfusion.10e.w_zh_en_ptt.pq.fifu").unwrap());
//     let embeds = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut ff_reader).unwrap();
//     let embedder = sentence_embedding::EmbeddingFetcher::new_from(tfidf, embeds);
//     let cutter = jieba_rs::Jieba::new();
//     let words_a = cutter
//         .cut("台灣總統", true)
//         .iter()
//         .map(|x| String::from(*x))
//         .collect::<Vec<String>>();
//     let a = embedder.words_to_vector(words_a).unwrap();
//     let words_b = cutter
//         .cut("臺灣總統", true)
//         .iter()
//         .map(|x| String::from(*x))
//         .collect::<Vec<String>>();
//     let b = embedder.words_to_vector(words_b).unwrap();
//     println!("{:?} - {:?} = {}", a, b, math_tool::consine_similarity(&a, &b).unwrap());
// }
#[derive(Debug)]
struct ReplyGroupWithSimilarity {
    reply_group: database::raw_ptt_article::CompressedReplies,
    similarity: f32,
}
struct Vec2Seq<'a> {
    embedder: sentence_embedding::EmbeddingFetcher,
    reply_group: granne::Granne<'a, angular::Vectors<'a>>,
    reply: granne::Granne<'a, angular::Vectors<'a>>,
    reply_group_database: leveldb::database::Database<i32>,
    cutter: jieba_rs::Jieba,
}
impl Vec2Seq<'_> {
    pub fn new(
        word_embedding: &std::path::Path,
        tfidf: &std::path::Path,
        stopwords: &std::path::Path,
        reply_group_index: &std::path::Path,
        reply_index: &std::path::Path,
        reply_group_elements: &std::path::Path,
        reply_elements: &std::path::Path,
        reply_group_database: &std::path::Path,
    ) -> Self {
        let mut options = leveldb::options::Options::new();
        options.create_if_missing = false;
        let mut _reply_group_database: leveldb::database::Database<i32> =
            match leveldb::database::Database::open(reply_group_database, options) {
                Ok(db) => db,
                Err(e) => panic!("failed to open database: {:?}", e),
            };
        let mut tfidf_bufreader = BufReader::new(File::open(tfidf).unwrap());
        let mut buf: Vec<u8> = Vec::new();
        tfidf_bufreader.read_to_end(&mut buf);
        let tfidf = bincode::deserialize::<tfidf::TfIdf>(&buf).unwrap();
        let mut ff_reader =
            BufReader::new(File::open("../cut_corpus/finalfusion.2e.w_zh.pq.fifu").unwrap());
        let embeds = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut ff_reader).unwrap();
        let mut stopwords_file = File::open("stopwords.txt").unwrap();
        let mut stopwords = fnv::FnvHashSet::default();
        let mut stop_words: String = String::new();
        stopwords_file.read_to_string(&mut stop_words);
        for word in stop_words.split("\n") {
            stopwords.insert(String::from(word));
        }
        let embedder = sentence_embedding::EmbeddingFetcher::new_from(tfidf, embeds, stopwords);
        let _reply_group = {
            let elements_file = File::open(reply_group_elements).unwrap();
            let index_file = File::open(reply_group_index).unwrap();
            let elements = unsafe { angular::Vectors::from_file(&elements_file).unwrap() };
            let index = unsafe { Granne::from_file(&index_file, elements).unwrap() };
            index
        };
        let _reply = {
            let elements_file = File::open(reply_group_elements).unwrap();
            let index_file = File::open(reply_group_index).unwrap();
            let elements = unsafe { angular::Vectors::from_file(&elements_file).unwrap() };
            let index = unsafe { Granne::from_file(&index_file, elements).unwrap() };
            index
        };
        Vec2Seq {
            embedder: embedder,
            reply_group: _reply_group,
            reply: _reply,
            reply_group_database: _reply_group_database,
            cutter: jieba_rs::Jieba::new(),
        }
    }
    pub fn search_replies(&self, sentence: String) -> Vec<ReplyGroupWithSimilarity> {
        let words = self
            .cutter
            .cut(&sentence[..], true)
            .iter()
            .map(|x| String::from(*x))
            .collect::<Vec<String>>();
        println!("{:?}", words);
        let vec = angular::Vector::from(self.embedder.words_to_vector(words).unwrap());
        let mut replies: Vec<ReplyGroupWithSimilarity> = Vec::new();
        {
            let mut ids = self.reply_group.search(&vec, 200, 10);
            replies.append(
                &mut ids[0..3].iter()
                    .map(|id| {
                        let read_opts = leveldb::options::ReadOptions::new();
                        let content = self
                            .reply_group_database
                            .get(read_opts, id.0 as i32)
                            .unwrap()
                            .unwrap();
                        ReplyGroupWithSimilarity {
                            reply_group: bincode::deserialize::<
                                database::raw_ptt_article::CompressedReplies,
                            >(&content)
                            .unwrap(),
                            similarity: id.1,
                        }
                    })
                    .collect::<Vec<ReplyGroupWithSimilarity>>(),
            );
        }
        {
            let mut ids = self.reply.search(&vec, 200, 10);
            replies.append(
                &mut ids[0..3].iter()
                    .map(|id| {
                        let read_opts = leveldb::options::ReadOptions::new();
                        let content = self
                            .reply_group_database
                            .get(read_opts, id.0 as i32)
                            .unwrap()
                            .unwrap();
                        ReplyGroupWithSimilarity {
                            reply_group: bincode::deserialize::<
                                database::raw_ptt_article::CompressedReplies,
                            >(&content)
                            .unwrap(),
                            similarity: id.1,
                        }
                    })
                    .collect::<Vec<ReplyGroupWithSimilarity>>(),
            );
        }
        replies
    }
}
fn test_vec2seq() {
    //create_self
    let vec2seq = Vec2Seq::new(
        std::path::Path::new("../cut_corpus/finalfusion.10e.w_zh_en_ptt.s60.pq.fifu"),
        std::path::Path::new("tfidf.bin"),
        std::path::Path::new("stopwords.txt"),
        std::path::Path::new("reply_group.index.granne"),
        std::path::Path::new("reply.index.granne"),
        std::path::Path::new("reply_group.element.granne"),
        std::path::Path::new("reply.element.granne"),
        std::path::Path::new("db/reply_group"),
    );
    let mut input: String = String::new();
    while true {
        //input sentence
        print!("Please enter some text: ");
        let _ = std::io::stdout().flush();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Did not enter a correct string");
        if let Some('\n') = input.chars().next_back() {
            input.pop();
        }
        if let Some('\r') = input.chars().next_back() {
            input.pop();
        }
        //return replies
        println!("{}", &input[..]);
        for reply in vec2seq.search_replies(input.clone()) {
            println!("{}: \n{:?}", reply.similarity, reply.reply_group.get());
        }
        input = String::new();
    }
}
fn main() {
    // tasks::rawarticle_to_tfidf();
    // tasks::rawarticle_filter_to_content_reply();
    // tasks::content_reply_to_reply_and_index();
    // tasks::test_reply_storage();
    // test_sentence_sim();
    test_vec2seq();
}
