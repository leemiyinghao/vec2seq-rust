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

#[derive(Debug)]
pub struct ReplyGroupWithSimilarity {
    pub reply_group: database::raw_ptt_article::CompressedReplies,
    pub similarity: f32,
}
pub struct Vec2Seq<'a> {
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
        let read_opts = leveldb::options::ReadOptions::new();
        _reply_group_database.get(read_opts, 0).unwrap().unwrap();
        let mut tfidf_bufreader = BufReader::new(File::open(tfidf).unwrap());
        let mut buf: Vec<u8> = Vec::new();
        tfidf_bufreader.read_to_end(&mut buf);
        let tfidf = bincode::deserialize::<tfidf::TfIdf>(&buf).unwrap();
        let mut ff_reader = BufReader::new(File::open(word_embedding).unwrap());
        let embeds = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut ff_reader).unwrap();
        let mut stopwords_file = File::open(stopwords).unwrap();
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
        _reply_group.get_element(0usize);
        let _reply = {
            let elements_file = File::open(reply_group_elements).unwrap();
            let index_file = File::open(reply_group_index).unwrap();
            let elements = unsafe { angular::Vectors::from_file(&elements_file).unwrap() };
            let index = unsafe { Granne::from_file(&index_file, elements).unwrap() };
            index
        };
        _reply.get_element(0usize);
        Vec2Seq {
            embedder: embedder,
            reply_group: _reply_group,
            reply: _reply,
            reply_group_database: _reply_group_database,
            cutter: jieba_rs::Jieba::new(),
        }
    }
    pub fn search_replies(
        &self,
        sentence: String,
        cascading: bool,
        threshold: f32,
        self_compare: Option<f32>,
    ) -> Option<Vec<String>> {
        //setting
        let self_pun = 0.1f32;
        let step = 0.05f32;
        let words = self
            .cutter
            .cut(&sentence[..], true)
            .iter()
            .map(|x| String::from(*x))
            .collect::<Vec<String>>();
        // println!("{:?}", words);
        let _vec = match self.embedder.words_to_vector(words) {
            Some(x) => x,
            None => return None,
        };
        let vec = angular::Vector::from(_vec.clone());
        let mut replies: Vec<ReplyGroupWithSimilarity> = Vec::new();
        let mut ids = self.reply_group.search(&vec, 200, 10);
        let mut self_ids = self.reply.search(&vec, 200, 10);
        let mut tmp: Vec<(usize, f32)> = Vec::new();
        if self_compare.is_some() {
            let __vec = &_vec.to_owned();
            for i in 0..ids.len() {
                ids[i].1 = ids[i].1 * (1f32 - self_compare.unwrap())
                    + match math_tool::consine_similarity(
                        &self.reply.get_element(ids[i].0).into_vec(),
                        &__vec,
                    ) {
                        Ok(x) => x,
                        Err(_) => 0f32,
                    } * self_compare.unwrap();
            }
        }
        for mut id in self_ids {
            let mut matched = false;
            for i in 0..ids.len() {
                if ids[i].0 == id.0 {
                    ids[i].1 = if self_compare.is_none() && id.1 - self_pun > ids[i].1 {
                        id.1
                    } else {
                        ids[i].1
                    };
                    matched = true;
                    break;
                } else {
                    continue;
                };
            }
            if !matched {
                id.1 = id.1 - self_pun;
                ids.push(id);
            };
        }
        //cascading
        if cascading {
            let mut max = 0f32;
            for id in ids.clone() {
                max = if id.1 > max { id.1 } else { max };
            }
            ids = ids
                .iter()
                .filter(|x| x.1 > (max - step))
                .map(|x| x.clone())
                .collect::<Vec<(usize, f32)>>();
        }
        ids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        {
            replies.append(
                &mut ids
                    .iter()
                    .map(|id| {
                        let read_opts = leveldb::options::ReadOptions::new();
                        //avoid empty reply_group
                        match self.reply_group_database.get(read_opts, id.0 as i32) {
                            Ok(x) => match x {
                                Some(content) => Some(ReplyGroupWithSimilarity {
                                    reply_group: bincode::deserialize::<
                                        database::raw_ptt_article::CompressedReplies,
                                    >(&content)
                                    .unwrap(),
                                    similarity: 1f32 - id.1,
                                }),
                                None => None,
                            },
                            Err(_) => None,
                        }
                    })
                    .filter(|x| x.is_some())
                    .map(|x| x.unwrap())
                    .collect::<Vec<ReplyGroupWithSimilarity>>(),
            );
        }
        let mut _replies: Vec<String> = Vec::new();
        for reply in replies {
            if reply.similarity > threshold {
                for text in reply.reply_group.get() {
                    _replies.push(text);
                }
            }
        }
        if _replies.len() > 0 {
            Some(_replies)
        } else {
            None
        }
    }
}
