use rusqlite::{params, Connection, Result};

pub mod filtered;
pub mod raw_ptt_article;

// pub fn get_raw_articles<'c>(
//     limit: &'c i32,
//     conn: &'c mut rusqlite::Connection,
// ) -> impl Iterator<Item = Result<raw_ptt_article::RawArticle>> + 'c {
//     //conn.execute("PRAGMA journal_mode=WAL", vec![]).unwrap();

// }
