package main

import (
  "log"
  "net"
  "fmt"
  "io"
  "os"
  "time"
)


type Config struct {
  Name string
  Port int
}

func ConnDispatch(conn net.Conn, config *Config) {
  log.Printf("%s: Local: %v Remote: %v", config.Name, conn.LocalAddr(), conn.RemoteAddr())
  go io.Copy(os.Stdout, conn)

  ticks := time.Tick(1 * time.Second)
  for t := range ticks {
    conn.Write([]byte(fmt.Sprintf("%s\n", t)))
  }
}


func main() {
  config := &Config{
    Name:   "Ticker",
    Port:   3751,
  }
  ln, err := net.Listen("tcp4", fmt.Sprintf(":%d", config.Port))
  if err != nil {
    log.Fatal(fmt.Sprintf("Create listener failed: %v", err))
  }
  defer ln.Close()

  for {
    conn, err := ln.Accept()
    if err != nil {
      log.Fatal(fmt.Sprintf("Accept connection failed: %v", err))
    }
    defer conn.Close()
    go ConnDispatch(conn, config)
  }
}
