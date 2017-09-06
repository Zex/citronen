package main

import (
  "net"
  "log"
  "fmt"
  "time"
  "io"
  "os"
)

func create_conn(port int) {
  conn, err := net.Dial("tcp4", fmt.Sprintf(":%d", port))
  if err != nil {
    log.Fatal(fmt.Sprintf("Dial remote failed: %v", err))
  }
  defer conn.Close()

  go func() {
    io.Copy(os.Stdout, conn)
  }()

  conn.Write([]byte(fmt.Sprintf("%s\n", time.Now())))
  ticks := time.Tick(1 * time.Second)
  for t := range ticks {
    conn.Write([]byte(fmt.Sprintf("%s\n", t)))
  }
}

func main() {
  port := 3751
  total, max_conn := 0, 10000

  for {
    go create_conn(port)
    total += 1
    if total > max_conn {
      break
    }
  }
  ticks := time.Tick(1 * time.Second)
  for _ = range ticks {}
}
