use fnv::FnvHashMap;
use jieba_rs;
use lz_fnv;
use lz_fnv::FnvHasher;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct TfIdf {
    pub doc_count: i64,
    pub matrix: Vec<u32>,
    #[serde(skip_serializing, skip_deserializing)]
    cutter: Option<jieba_rs::Jieba>,
}
impl TfIdf {
    pub fn new() -> Self {
        TfIdf {
            doc_count: 0,
            matrix: vec![0u32; 100000],
            cutter: None,
        }
    }
    pub fn new_from_cutter(cutter: jieba_rs::Jieba) -> Self {
        TfIdf {
            doc_count: 0,
            matrix: vec![0u32; 100000],
            cutter: Some(cutter),
        }
    }
    pub fn add_uncut_document(&mut self, content: String) -> Result<(), ()> {
        if self.cutter.is_none() {
            self.cutter = Some(jieba_rs::Jieba::new());
        }
        let _content = self
            .cutter
            .as_ref()
            .unwrap()
            .cut(&content[..], true)
            .iter()
            .map(|x| String::from(*x))
            .collect::<Vec<String>>();
        self.add_cutted_document(_content)
    }
    pub fn add_cutted_document(&mut self, content: Vec<String>) -> Result<(), ()> {
        let mut tmp: FnvHashMap<String, bool> = FnvHashMap::default();
        let mut hasher = lz_fnv::Fnv1a::<u32>::new();
        self.doc_count += 1;
        for word in content {
            if !tmp.contains_key(&word) {
                tmp.insert(word.clone(), true);
                hasher.write(&word.as_bytes());
                let key = hasher.finish() % 100000;
                self.matrix[key as usize] += 1;
            };
        }
        Ok(())
    }
    pub fn get_log_idf(&self, word: String) -> f32 {
        //log(doc_count / (1 + df_of_word))
        let mut hasher = lz_fnv::Fnv1a::<u32>::new();
        hasher.write(&word.as_bytes());
        let key = hasher.finish() % 100000;
        let df_of_word = self.matrix[key as usize];
        // println!("df_of_word: {}, doc_count: {}", df_of_word, self.doc_count);
        ((self.doc_count / (1 + df_of_word as i64)) as f32).log2()
    }
    pub fn get_max(&self) -> u32 {
        let mut max: u32 = 0;
        for i in self.matrix.iter() {
            if *i > max {
                max = *i;
            };
        }
        max
    }
}
