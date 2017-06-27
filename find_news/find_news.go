package main

import (
    "github.com/balzaczyy/golucene/core/index"
    "github.com/balzaczyy/golucene/core/search"
    "github.com/balzaczyy/golucene/core/store"
    "github.com/balzaczyy/golucene/core/util"
    "github.com/balzaczyy/golucene/core/document"
    "github.com/balzaczyy/golucene/analysis/standard"
    _ "github.com/balzaczyy/golucene/core/codec/lucene410"
    "flag"
    "os"
    "log"
    "bufio"
    "io/ioutil"
    "path/filepath"
)

func build_index(index_base string, src_base string) (ret bool){
    os.MkdirAll(index_base, 0777)
    dir, err := store.OpenFSDirectory(index_base)
    if err != nil {
      log.Fatalln("open directory failed:", index_base)
      return false
    }
    defer dir.Close()

    index.DefaultSimilarity = func() index.Similarity {
      return search.NewDefaultSimilarity()
    }
    analyzer := standard.NewStandardAnalyzer()
    cfg := index.NewIndexWriterConfig(util.VERSION_LATEST, analyzer)
    iw, err := index.NewIndexWriter(dir, cfg)
    if err != nil {
      log.Fatalln("create index writer failed", err)
      return false
    }
    defer iw.Close()

    src_paths, _ := store.FSDirectoryListAll(src_base)

    for i, path := range src_paths {
      log.Println(i, path)
      doc := get_doc(path)
      if doc != nil {
        err := iw.AddDocument(doc.Fields())
        if err != nil {
          log.Fatalln("add doc failed", err)
          break
        }
      } else {
        log.Fatalln("doc is nil")
        break
      }
    }

    return true
}

func get_doc(path string) (doc *document.Document) {
    doc = document.NewDocument()
    full_path := filepath.Join(*src_base, path)
    content, err := ioutil.ReadFile(full_path)
    if err != nil {
      log.Fatalln("read file failed:", path, err)
      return nil
    }
    doc.Add(document.NewTextFieldFromString("name", path, document.STORE_YES))
    doc.Add(document.NewTextFieldFromString("path", full_path, document.STORE_YES))
    doc.Add(document.NewTextFieldFromString("content", string(content), document.STORE_YES))
    return doc
}

func init() {
    println("=================================")
}

func do_search(index_base string, term string, query string) (ret bool){
    dir, err := store.OpenFSDirectory(index_base)
    if err != nil {
      log.Fatalln("open directory failed: ", index_base, err)
      return false
    }
    defer dir.Close()

    ir, err := index.OpenDirectoryReader(dir)
    if err != nil {
      log.Fatalln("open dir reader failed", err)
      return false
    }
    defer ir.Close()

    searcher := search.NewIndexSearcher(ir)
    q := search.NewTermQuery(index.NewTerm(term, query))
    if searcher == nil {
      log.Fatalln("create search query failed")
      return false
    }
    docs, err := searcher.Search(q, nil, 5)
    if err != nil {
      log.Fatalln("search failed: ", query, err)
      return false
    }

    log.Println("total hists:", docs.TotalHits)
    for _, doc := range docs.ScoreDocs {
      log.Println("hit: ", doc, doc.Doc)
      explain, err := searcher.Explain(q, doc.Doc)
      if err != nil {
        log.Fatalln("explain failed", err)
        continue
      }
      log.Println("how: ", explain)
    }
    return true
}

func show_paths(paths []string) {
    for i, path := range paths {
      log.Println(i, path)
    }
}

var (
    pwd, _ = os.Getwd()
    content = flag.String("content", "", "Search content with keyword")
    name = flag.String("name", "", "Search name with keyword")
    path = flag.String("path", "", "Search path with keyword")
    src_base = flag.String("src_base", "", "Path to dataset")
    index_base = flag.String("index_base", "", "Path to index")
    prefix = flag.String("prefix", "oov", "Prefix for logging")
    logger = log.New(bufio.NewWriter(os.Stdout), *prefix, log.LstdFlags|log.Lshortfile)
)

func main() {
    flag.Parse()
    if len(*index_base) > 0 && len(*src_base) > 0 {
      build_index(*index_base, *src_base)
    }
    if len(*index_base) > 0{
      if len(*content) > 0 {
        do_search(*index_base, "content", *content)
      }
      if len(*name) > 0 {
        do_search(*index_base, "name", *name)
      }
      if len(*path) > 0 {
        do_search(*index_base, "path", *path)
      }
    }
}
