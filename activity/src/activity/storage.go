package activity

import (
  "fmt"
  "database/sql"
  "log"
  _ "gopkg.in/cq.v1"
)

var (
  cred = "neo4j:j5NEO"
  host = "localhost"
  port = 7474
)

func Insert(partners Partners) error {
  conn, err := sql.Open("neo4j-cypher", fmt.Sprintf("http://%s@%s:%d", cred, host, port))
  if err != nil {
    log.Fatal(err)
    return err
  }
  defer conn.Close()

  tran, err := conn.Begin()
  if err != nil {
    log.Fatal(err)
    return err
  }

  stmt, err := conn.Prepare(`
    create (n:partner {name:{0}, level:{1}, created:{2}})
  `)

  if err != nil {
    log.Fatal(err)
    return err
  }

  for _, partner := range(partners) {
    stmt.Exec(partner.Name, partner.Level, partner.Created)
  }

  err = tran.Commit()
  if err != nil {
    log.Fatal(err)
    return err
  }
  return nil
}

