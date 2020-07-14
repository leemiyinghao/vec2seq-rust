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
use vec2seq_rust::Vec2Seq;
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
    loop {
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
        for reply in match vec2seq.search_replies(input.clone(), true) {
            Some(x) => x,
            None => continue,
        } {
            println!("{}", reply);
        }
        input = String::new();
    }
}
use jieba_rs::{Jieba, TokenizeMode};
fn jieba_dag_test() {
    let j = Jieba::new();
    println!(
        "{:?}",
        j.tokenize(
            "有沒有新竹美食的八卦?在這邊快餓死了",
            TokenizeMode::Default,
            true
        )
    );
}
fn main() {
    // tasks::raw_article_sqlite_to_leveldb();
    // tasks::rawarticle_to_tfidf();
    // tasks::rawarticle_filter_to_content_reply();
    // tasks::content_reply_to_reply_and_index();
    // tasks::test_reply_storage();
    // test_sentence_sim();
    test_vec2seq();
    // jieba_dag_test();
}
