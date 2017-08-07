package activity

import (
  "fmt"
  "database/sql"
  "log"
  _ "gopkg.in/cq.v1"
)

func Insert(partners Partners) error {
  conn, err := sql.Open("neo4j-cypher", "http://neo4j:j5NEO@localhost:7474")
  if err != nil {
    log.Fatal(err)
    return err
  }

  tran, err := conn.Begin()
  if err != nil {
    log.Fatal(err)
    return err
  }

  stmt, err := conn.Prepare("create (:partner {name:{0}, level:{1}, created:{2}})")

  if err != nil {
    log.Fatal(err)
    return err
  }

  for _, partner := range(partners) {
    fmt.Println(partner)
    stmt.Exec(partner.Name, partner.Level, partner.Created)
  }

  err = tran.Commit()
  if err != nil {
    log.Fatal(err)
    return err
  }
  return nil
}

