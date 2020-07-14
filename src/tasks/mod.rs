use crate::database;
use crate::math_tool;
use crate::sentence_embedding;
use crate::tfidf;
use crossbeam::crossbeam_channel::{bounded, unbounded};
use finalfusion::prelude::*;
use indicatif::{HumanDuration, MultiProgress, ProgressBar, ProgressStyle};
use jieba_rs;
use leveldb;
use leveldb::database::iterator::Iterable;
use leveldb::kv::KV;
use leveldb_sys;
use rand::Rng;
use regex::Regex;
use rusqlite::{params, Connection, Result};
use serde_json::Value;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use threadpool::ThreadPool;
pub fn raw_article_sqlite_to_leveldb() {
    let conn = Connection::open("../wakeupscrew_line/utility/pttRawStage1.sqlite").unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT id, title, content, board, pushes, used FROM rawarticle WHERE pushes<>'[]' and pushes<>''",
        )
        .unwrap();
    let mut articles = stmt
        .query_map(params![], move |row| {
            Ok(database::raw_ptt_article::RawArticle {
                id: row.get(0)?,
                title: row.get(1)?,
                content: row.get(2)?,
                board: row.get(3)?,
                pushes: row.get(4)?,
                used: row.get(5)?,
            })
        })
        .unwrap();
    let mut options = leveldb::options::Options::new();
    options.create_if_missing = true;
    options.compression = leveldb_sys::Compression::Snappy;
    let mut database = match leveldb::database::Database::open(Path::new("db/ldb.ldb"), options) {
        Ok(db) => db,
        Err(e) => panic!("failed to open database: {:?}", e),
    };
    let count = 2730635;
    let pb = ProgressBar::new(count);
    pb.set_style(ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{bar:cyan/blue}] {pos}/{len} {per_sec} ({eta})",
    ));
    let mut step = 1;
    for curr in articles {
        let _curr = curr.unwrap();
        let im_article = database::raw_ptt_article::Article {
            origin: database::raw_ptt_article::ArticleOrigin::PttOrigin {
                title: _curr.title.clone(),
                board: _curr.board.clone(),
            },
            content: String::from(format!(
                "{}\n{}",
                _curr.title.clone(),
                _curr.content.clone()
            )),
            pushes: {
                let pushes: Vec<database::raw_ptt_article::Push> =
                    serde_json::from_str::<Vec<database::raw_ptt_article::Push>>(&_curr.pushes)
                        .unwrap();
                let mut reduced_pushes: Vec<database::raw_ptt_article::Push> = Vec::new();
                for push in pushes {
                    if reduced_pushes.len() == 0
                        || reduced_pushes.last().unwrap().push_userid != push.push_userid
                    {
                        reduced_pushes.push(push);
                    } else {
                        reduced_pushes.last_mut().unwrap().push_content += &push.push_content;
                    }
                }
                reduced_pushes.into_iter().map(|x| x.push_content).collect()
            },
        };
        let write_opts = leveldb::options::WriteOptions::new();
        match database.put(
            write_opts,
            step,
            bincode::serialize(&im_article).unwrap().as_ref(),
        ) {
            Ok(_) => (),
            Err(e) => panic!("failed to write to database: {:?}", e),
        };
        step += 1;
        pb.inc(1);
    }
}
struct KMediumObject {
    embedding: Vec<f32>,
    context: String,
}
struct KMediumGroup {
    medium: Vec<f32>,
    objects: Vec<KMediumObject>,
}
impl KMediumGroup {
    pub fn updateMedium(&mut self) {
        if (self.objects.len() > 0) {
            let width = self.objects[0].embedding.len();
            let mut sum: Vec<f32> = vec![0f32; width];
            for obj in self.objects.iter() {
                for i in 0..width {
                    sum[i] += obj.embedding[i] / (self.objects.len() as f32);
                }
            }
            self.medium = sum;
        }
    }
}
pub fn rawarticle_filter_to_content_reply() {
    let mut options = leveldb::options::Options::new();
    options.create_if_missing = true;
    options.compression = leveldb_sys::Compression::Snappy;
    let mut input_database: leveldb::database::Database<i32> =
        match leveldb::database::Database::open(Path::new("db/ldb.ldb"), options) {
            Ok(db) => db,
            Err(e) => panic!("failed to open database: {:?}", e),
        };
    let mut options = leveldb::options::Options::new();
    options.create_if_missing = true;
    options.compression = leveldb_sys::Compression::Snappy;
    let mut output_database: leveldb::database::Database<i32> =
        match leveldb::database::Database::open(Path::new("db/content_reply"), options) {
            Ok(db) => db,
            Err(e) => panic!("failed to open database: {:?}", e),
        };
    // let mut options = leveldb::options::Options::new();
    // options.create_if_missing = true;
    // options.compression = leveldb_sys::Compression::Snappy;
    // let mut reply_output_database: leveldb::database::Database<i32> =
    //     match leveldb::database::Database::open(Path::new("db/reply"), options) {
    //         Ok(db) => db,
    //         Err(e) => panic!("failed to open database: {:?}", e),
    //     };
    let read_opts = leveldb::options::ReadOptions::new();
    let iter = input_database.iter(read_opts);
    let mut step = 0i32;
    let pb = ProgressBar::new(2655917);
    pb.set_style(ProgressStyle::default_bar().template(
        "processing: [{elapsed_precise}] [{bar:cyan/blue}] {pos}/{len} {per_sec} ({eta})",
    ));
    let workers = 12 - 2;
    let (work_sender, work_receiver) =
        bounded::<Result<database::raw_ptt_article::Article, ()>>(workers);
    let (result_sender, result_receiver) =
        bounded::<Result<Vec<database::raw_ptt_article::ContentReply>, ()>>(workers);
    let (join_sender, join_receiver) = bounded::<()>(workers);
    let mut joins: Vec<std::thread::JoinHandle<()>> = Vec::new();
    //establish workers
    for i in 0..workers {
        let clone_work_receiver = work_receiver.clone();
        let clone_result_sender = result_sender.clone();
        let clone_join_sender = join_sender.clone();
        let handle = std::thread::spawn(move || {
            let mut tfidf_bufreader = BufReader::new(File::open("tfidf.bin").unwrap());
            let mut buf: Vec<u8> = Vec::new();
            tfidf_bufreader.read_to_end(&mut buf);
            let tfidf = bincode::deserialize::<tfidf::TfIdf>(&buf).unwrap();
            let mut ff_reader = BufReader::new(
                File::open("../cut_corpus/finalfusion.10e.w_zh_en_ptt.s60.pq.fifu").unwrap(),
            );
            let embeds =
                Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut ff_reader).unwrap();
            let mut stopwords_file = File::open("stopwords.txt").unwrap();
            let mut stopwords = fnv::FnvHashSet::default();
            let mut stop_words: String = String::new();
            stopwords_file.read_to_string(&mut stop_words);
            for word in stop_words.split("\n") {
                stopwords.insert(String::from(word));
            }
            let embedder = sentence_embedding::EmbeddingFetcher::new_from(tfidf, embeds, stopwords);
            let cutter = jieba_rs::Jieba::new();
            let link_rule = Regex::new(r#"https?.+"#).unwrap();
            let sentence_rule = Regex::new(r#"(\?|？|!|！|ww+|。)"#).unwrap();
            let url_rule = Regex::new(r#"https?://[^\s]+"#).unwrap();
            let unmeanful_rule = Regex::new(r#"^XD+|推$"#).unwrap();
            while true {
                let _article = match clone_work_receiver.recv().unwrap() {
                    Ok(x) => x,
                    Err(_) => break,
                };

                // let mut filtered = database::raw_ptt_article::ContentReply {
                //     content: _article.content,
                //     replies: Vec::new(),
                // };
                let book_rule = Regex::new(r#"\[.+\].+"#).unwrap();
                //filter reply
                let mut book_replies: Vec<String> = Vec::new();
                match _article.origin {
                    database::raw_ptt_article::ArticleOrigin::PttOrigin { title, board } => {
                        if board == "AC_In" {
                            for push in &_article.pushes {
                                if book_rule.is_match(&push[..]) {
                                    book_replies.push(push.clone());
                                }
                            }
                        }
                    }
                    _ => {}
                };
                let mut groups: Vec<KMediumGroup> = Vec::new();

                for push in _article.pushes {
                    //skip links
                    if link_rule.is_match(&push.clone()[..]) {
                        continue;
                    }
                    if unmeanful_rule.is_match(&push.clone()[..]) {
                        continue;
                    }
                    let hit = false;
                    let words = cutter
                        .cut(&push.clone()[..], true)
                        .iter()
                        .map(|x| String::from(*x))
                        .collect::<Vec<String>>();
                    let embedding = match embedder.words_to_vector(words) {
                        Some(x) => x,
                        None => continue,
                    };
                    let mut closest_medium: Option<usize> = None;
                    let mut closest_sim = 0f32;
                    for i in 0..groups.len() {
                        // let sim = math_tool::consine_similarity(&groups[i].medium, &embedding)
                        // .unwrap_or(0f32);
                        let sim =
                            math_tool::consine_similarity(&groups[i].medium, &embedding).unwrap(); //see what will happen
                        if sim > 0.85f32 && sim > closest_sim {
                            closest_medium = Some(i);
                            closest_sim = sim;
                        }
                    }
                    match closest_medium {
                        Some(i) => {
                            groups[i].objects.push(KMediumObject {
                                context: push.clone(),
                                embedding: embedding,
                            });
                            groups[i].updateMedium();
                        }
                        None => {
                            let mut new_group = KMediumGroup {
                                medium: Vec::new(),
                                objects: vec![KMediumObject {
                                    context: push.clone(),
                                    embedding: embedding,
                                }],
                            };
                            new_group.updateMedium();
                            groups.push(new_group);
                        }
                    }
                }
                let mut __articles: Vec<database::raw_ptt_article::ContentReply> = sentence_rule
                    .split(&url_rule.replace(&_article.content.clone()[..], "")[..])
                    .map(|sentence| database::raw_ptt_article::ContentReply {
                        content: String::from(sentence),
                        replies: Vec::new(),
                    })
                    .collect();
                __articles[0].replies.append(&mut book_replies);
                for group in groups {
                    if group.objects.len() > 5 {
                        let mut closest = 0;
                        let mut sim = 0f32;
                        for i in 0..__articles.len() {
                            let words = cutter
                                .cut(&__articles[i].content.clone()[..], true)
                                .iter()
                                .map(|x| String::from(*x))
                                .collect::<Vec<String>>();
                            let embedding = match embedder.words_to_vector(words) {
                                Some(x) => x,
                                None => continue,
                            };
                            let _sim =
                                math_tool::consine_similarity(&group.medium, &embedding).unwrap();
                            if _sim > sim {
                                sim = _sim;
                                closest = i;
                            }
                        }
                        let mut filtered_rerplies: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        for obj in group.objects {
                            filtered_rerplies.insert(obj.context);
                        }
                        for key in filtered_rerplies {
                            __articles[0].replies.push(key.clone());
                            if closest != 0 {
                                __articles[closest].replies.push(key);
                            }
                        }
                    }
                }
                clone_result_sender.send(Ok(__articles)).unwrap();
            }
            clone_join_sender.send(()).unwrap();
        });
        joins.push(handle);
    }
    //result writer
    joins.push(std::thread::spawn(move || {
        let mut step = 0i32;
        let mut reply_step = 0i32;
        while true {
            let __articles = match result_receiver.recv().unwrap() {
                Ok(x) => x,
                Err(_) => break,
            };
            for result in __articles {
                if result.replies.len() > 0 {
                    //break into sections, pairing group, multi write
                    let write_opts = leveldb::options::WriteOptions::new();
                    match output_database.put(
                        write_opts,
                        step,
                        bincode::serialize(&result).unwrap().as_ref(),
                    ) {
                        Ok(_) => (),
                        Err(e) => panic!("failed to write to database: {:?}", e),
                    };
                    step += 1;
                }
            }
        }
        println!("\nwrite {} records", step);
        join_sender.send(()).unwrap();
    }));
    for article in iter {
        pb.inc(1);
        let _article = match bincode::deserialize::<database::raw_ptt_article::Article>(&article.1)
        {
            Ok(a) => a,
            Err(_) => continue,
        };
        work_sender.send(Ok(_article)).unwrap();
    }
    //QA Set Loading
    let qa_set = BufReader::new(File::open("db/Gossiping-QA-Dataset-2_0.csv").unwrap()).lines();
    let mut line_num = 0;
    for line in qa_set {
        if line_num != 0 {
            let _line = match line {
                Ok(x) => x,
                Err(x) => break,
            };
            let mut qa = _line.split("\t");
            let question = match qa.next() {
                Some(x) => x,
                None => continue,
            };
            let answer = match qa.next() {
                Some(x) => x,
                None => continue,
            };
            let mut article = database::raw_ptt_article::ContentReply {
                content: String::from(question),
                replies: vec![String::from(answer)],
            };
            result_sender.send(Ok(vec![article])).unwrap();
        }
        line_num += 1;
    }
    println!("{} from qa_set", line_num - 1);
    for i in 0..workers {
        work_sender.send(Err(())).unwrap();
        join_receiver.recv().unwrap();
    }
    result_sender.send(Err(())).unwrap();
    join_receiver.recv().unwrap();
    //wait for threads join
    let mut _step = 0i32;
    for i in joins {
        i.join().unwrap();
        println!("thread {} joined", _step);
        _step += 1;
    }
}

pub fn rawarticle_to_tfidf() {
    let mut options = leveldb::options::Options::new();
    options.create_if_missing = true;
    options.compression = leveldb_sys::Compression::Snappy;
    let mut input_database: leveldb::database::Database<i32> =
        match leveldb::database::Database::open(Path::new("db/ldb.ldb"), options) {
            Ok(db) => db,
            Err(e) => panic!("failed to open database: {:?}", e),
        };
    let read_opts = leveldb::options::ReadOptions::new();
    let iter = input_database.iter(read_opts);
    let workers = 12 - 2;
    let (work_sender, work_receiver) = bounded::<Option<Vec<u8>>>(workers);
    let (result_sender, result_receiver) = bounded::<tfidf::TfIdf>(workers);
    for i in 0..workers {
        let clone_work_receiver = work_receiver.clone();
        let clone_result_sender = result_sender.clone();
        std::thread::spawn(move || {
            let mut tfidf_table = tfidf::TfIdf::new();
            while true {
                let work = clone_work_receiver.recv().unwrap();
                match work {
                    Some(x) => {
                        match bincode::deserialize::<database::raw_ptt_article::Article>(&x) {
                            Ok(a) => {
                                tfidf_table.add_uncut_document(a.content);
                                for push in a.pushes {
                                    tfidf_table.add_uncut_document(push);
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                    None => {
                        clone_result_sender.send(tfidf_table).unwrap();
                        break;
                    }
                };
            }
        });
    }
    let pb = ProgressBar::new(2655917);
    pb.set_style(ProgressStyle::default_bar().template(
        "processing: [{elapsed_precise}] [{bar:cyan/blue}] {pos}/{len} {per_sec} ({eta})",
    ));
    let mut tfidf_table = tfidf::TfIdf::new();
    for article in iter {
        pb.inc(1);
        work_sender.send(Some(article.1)).unwrap();
    }
    pb.finish_and_clear();
    for i in 0..workers {
        work_sender.send(None).unwrap();
        let partial = result_receiver.recv().unwrap();
        for j in 0..partial.matrix.len() {
            tfidf_table.matrix[j] += partial.matrix[j];
        }
        tfidf_table.doc_count += partial.doc_count;
    }
    println!("max df is: {}", tfidf_table.get_max());
    println!("doc_count is: {}", tfidf_table.doc_count);
    let mut file = File::create("tfidf.bin").unwrap();
    file.write_all(&bincode::serialize(&tfidf_table).unwrap());
}

use granne::{angular, BuildConfig, Builder, Granne, GranneBuilder, Index};
struct TextVectorSet {
    text: String,
    vector: Vec<f32>,
}
struct ArticleVectorSet {
    content: TextVectorSet,
    replies: Vec<TextVectorSet>,
}
pub fn content_reply_to_reply_and_index() {
    let mut options = leveldb::options::Options::new();
    options.create_if_missing = false;
    options.compression = leveldb_sys::Compression::Snappy;
    let mut input_database: leveldb::database::Database<i32> =
        match leveldb::database::Database::open(Path::new("db/content_reply"), options) {
            Ok(db) => db,
            Err(e) => panic!("failed to open database: {:?}", e),
        };
    //slice sentence vectors
    let workers = 12 - 2;
    let (work_sender, work_receiver) =
        bounded::<Option<database::raw_ptt_article::ContentReply>>(workers);
    let (result_sender, result_receiver) = bounded::<Option<ArticleVectorSet>>(workers);
    let (join_sender, join_receiver) = bounded::<()>(workers);

    //establish workers
    for i in 0..workers {
        let clone_work_receiver = work_receiver.clone();
        let clone_result_sender = result_sender.clone();
        let clone_join_sender = join_sender.clone();
        let handle = std::thread::spawn(move || {
            let mut tfidf_bufreader = BufReader::new(File::open("tfidf.bin").unwrap());
            let mut buf: Vec<u8> = Vec::new();
            tfidf_bufreader.read_to_end(&mut buf);
            let tfidf = bincode::deserialize::<tfidf::TfIdf>(&buf).unwrap();
            let mut ff_reader = BufReader::new(
                File::open("../cut_corpus/finalfusion.10e.w_zh_en_ptt.s60.pq.fifu").unwrap(),
            );
            let embeds =
                Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut ff_reader).unwrap();
            let mut stopwords_file = File::open("stopwords.txt").unwrap();
            let mut stopwords = fnv::FnvHashSet::default();
            let mut stop_words: String = String::new();
            stopwords_file.read_to_string(&mut stop_words);
            for word in stop_words.split("\n") {
                stopwords.insert(String::from(word));
            }
            let embedder = sentence_embedding::EmbeddingFetcher::new_from(tfidf, embeds, stopwords);
            let cutter = jieba_rs::Jieba::new();
            while true {
                let _article = match clone_work_receiver.recv().unwrap() {
                    Some(x) => x,
                    None => break,
                };
                let mut result = ArticleVectorSet {
                    content: TextVectorSet {
                        text: _article.content.clone(),
                        vector: match embedder.words_to_vector(
                            cutter
                                .cut(&_article.content[..], true)
                                .iter()
                                .map(|x| String::from(*x))
                                .collect::<Vec<String>>(),
                        ) {
                            Some(x) => x,
                            None => continue,
                        },
                    },
                    replies: _article
                        .replies
                        .iter()
                        .map(|reply| TextVectorSet {
                            text: reply.clone(),
                            vector: match embedder.words_to_vector(
                                cutter
                                    .cut(&reply[..], true)
                                    .iter()
                                    .map(|x| String::from(*x))
                                    .collect::<Vec<String>>(),
                            ) {
                                Some(x) => x,
                                None => Vec::new(),
                            },
                        })
                        .filter(|x| x.vector.len() > 0)
                        .collect(),
                };
                clone_result_sender.send(Some(result)).unwrap();
            }
            clone_result_sender.send(None).unwrap();
        });
    }
    //work reader
    std::thread::spawn(move || {
        let read_opts = leveldb::options::ReadOptions::new();
        let iter = input_database.iter(read_opts);
        let mut step = 0i32;
        for article in iter {
            let _article =
                match bincode::deserialize::<database::raw_ptt_article::ContentReply>(&article.1) {
                    Ok(a) => a,
                    Err(_) => continue,
                };
            work_sender.send(Some(_article)).unwrap();
        }
        for i in 0..workers {
            work_sender.send(None).unwrap();
        }
        println!("read {} records", step);
        // join_sender.send(()).unwrap();
    });
    // let mut content_texts: Vec<String> = Vec::new();
    let mut content_vectors = granne::angular::Vectors::new();
    // let mut reply_texts: Vec<String> = Vec::new();
    let mut reply_vectors = granne::angular::Vectors::new();
    let mut finished = 0usize;
    let pb = ProgressBar::new(1603811);
    pb.set_style(ProgressStyle::default_bar().template(
        "processing: [{elapsed_precise}] [{bar:cyan/blue}] {pos}/{len} {per_sec} ({eta})",
    ));
    let mut options = leveldb::options::Options::new();
    options.create_if_missing = true;
    // options.compression = leveldb_sys::Compression::Snappy;
    let mut reply_group_output_database: leveldb::database::Database<i32> =
        match leveldb::database::Database::open(Path::new("db/reply_group"), options) {
            Ok(db) => db,
            Err(e) => panic!("failed to open database: {:?}", e),
        };
    //result writer
    loop {
        pb.inc(1);
        if finished >= workers {
            break;
        }
        let result = match result_receiver.recv().unwrap() {
            Some(x) => x,
            None => {
                finished += 1;
                continue;
            }
        };
        let step = content_vectors.len() as i32;
        content_vectors.push(&angular::Vector::from(result.content.vector));
        let mut replies: Vec<String> = Vec::new();
        let mut reply_vector: Vec<Vec<f32>> = Vec::new();
        // let mut writers: Vec<std::thread::JoinHandle<()>> = Vec::new();
        for reply in result.replies {
            let write_opts = leveldb::options::WriteOptions::new();
            replies.push(reply.text);
            reply_vector.push(reply.vector);
        }
        let mut reply_vector_mean: Vec<f32> = vec![0f32; reply_vector[0].len()];
        let vector_len = reply_vector.len() as f32;
        for _r in reply_vector {
            for i in 0.._r.len() {
                reply_vector_mean[i] += _r[i] / vector_len;
            }
        }
        reply_vectors.push(&angular::Vector::from(reply_vector_mean));
        let _replies = database::raw_ptt_article::CompressedReplies::new_from(&replies);
        let write_opts = leveldb::options::WriteOptions::new();
        match reply_group_output_database.put(
            write_opts,
            step,
            bincode::serialize(&_replies).unwrap().as_ref(),
        ) {
            Ok(_) => (),
            Err(e) => panic!("failed to write to database: {:?}", e),
        };
    }
    pb.finish();

    println!();
    println!(
        "received {} articles, {} replies",
        content_vectors.len(),
        reply_vectors.len()
    );
    {
        //build content
        let build_config = granne::BuildConfig::default()
            .show_progress(true)
            .max_search(10);
        let mut builder = GranneBuilder::new(build_config, content_vectors);
        builder.build();

        // saving to disk
        let mut index_file = File::create("reply_group.index.granne").unwrap();
        builder.write_index(&mut index_file).unwrap();
        let mut elements_file = File::create("reply_group.element.granne").unwrap();
        builder.write_elements(&mut elements_file).unwrap();
    }
    {
        //build reply
        let build_config = granne::BuildConfig::default()
            .show_progress(true)
            .max_search(10);
        let mut builder = GranneBuilder::new(build_config, reply_vectors);
        builder.build();

        // saving to disk
        let mut index_file = File::create("reply.index.granne").unwrap();
        builder.write_index(&mut index_file).unwrap();
        let mut elements_file = File::create("reply.element.granne").unwrap();
        builder.write_elements(&mut elements_file).unwrap();
    }
}
pub fn test_reply_storage() {
    let mut options = leveldb::options::Options::new();
    options.create_if_missing = false;
    options.compression = leveldb_sys::Compression::Snappy;
    let mut input_database: leveldb::database::Database<i32> =
        match leveldb::database::Database::open(Path::new("db/content_reply"), options) {
            Ok(db) => db,
            Err(e) => panic!("failed to open database: {:?}", e),
        };
    let read_opts = leveldb::options::ReadOptions::new();
    let iter = input_database.iter(read_opts);
    let mut step = 0i32;
    let pb = ProgressBar::new(224669);
    pb.set_style(ProgressStyle::default_bar().template(
        "processing: [{elapsed_precise}] [{bar:cyan/blue}] {pos}/{len} {per_sec} ({eta})",
    ));
    let mut reply_groups: Vec<database::raw_ptt_article::CompressedReplies> = Vec::new();
    for article in iter {
        pb.inc(1);
        let _article =
            match bincode::deserialize::<database::raw_ptt_article::ContentReply>(&article.1) {
                Ok(a) => a,
                Err(_) => continue,
            };
        let mut replies: Vec<String> = Vec::new();
        for reply in _article.replies {
            replies.push(reply);
        }
        // reply_group_file.write_all(bincode::serialize(&reply_ids).unwrap().as_ref());
        let _replies = database::raw_ptt_article::CompressedReplies::new_from(&replies);
        // assert_eq!(_replies.get().len(), replies.len());
        // assert_eq!(_replies.get()[0], replies[0]);
        reply_groups.push(_replies);
    }
    pb.finish();
    println!("");
    println!("processed {} groups", reply_groups.len());
    let mut reply_group_file = File::create("reply_group.bin").unwrap();
    let mut reply_file = File::create("reply.bin").unwrap();
    reply_group_file
        .write_all(bincode::serialize(&reply_groups).unwrap().as_ref())
        .unwrap();
}
