use crate::tfidf;
use finalfusion::prelude::*;
use regex::Regex;

pub struct EmbeddingFetcher {
    word_embedding: Embeddings<VocabWrap, StorageWrap>,
    tfidf: tfidf::TfIdf,
    stopwords: fnv::FnvHashSet<String>,
    empty_checker: Regex,
}
impl EmbeddingFetcher {
    pub fn new_from(
        tfidf: tfidf::TfIdf,
        word_embedding: Embeddings<VocabWrap, StorageWrap>,
        stopwords: fnv::FnvHashSet<String>,
    ) -> Self {
        let empty_checker = Regex::new(r"^\s+$").unwrap();
        EmbeddingFetcher {
            tfidf,
            word_embedding,
            stopwords,
            empty_checker,
        }
    }
    pub fn word_to_vector(&self, word: String) -> Vec<f32> {
        let weight = self.tfidf.get_log_idf(word.clone());
        let embedding = self.word_embedding.embedding(&word[..]).unwrap();
        let mut tmp: Vec<f32> = Vec::new();
        for i in 0..embedding.len() {
            tmp.push(weight * embedding[i]);
        }
        tmp
    }
    pub fn words_to_vector(&self, words: Vec<String>) -> Option<Vec<f32>> {
        let vecs = words
            .iter()
            // .filter(|x| !self.empty_checker.is_match(x) && !self.stopwords.contains(x.clone()))
            .filter(|x| !self.empty_checker.is_match(x))
            .map(|x| self.word_to_vector((*x).clone()))
            .collect::<Vec<Vec<f32>>>();
        if vecs.len() == 0 {
            return None;
        }
        let len = vecs[0].len();
        let mut sum: Vec<f32> = vec![0f32; len];
        let vec_len = vecs.len() as f32;
        for vec in vecs {
            for i in 0..len {
                // println!("{:?}/{:?} = {:?}", vec[i], vec_len, vec[i] / vec_len);
                sum[i] += vec[i] / vec_len;
            }
            // println!("{:?}", sum);
        }
        Some(sum)
    }
}
