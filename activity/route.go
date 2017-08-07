package main

import (
  "fmt"
  "html"
  "log"
  "time"
  "net/http"
  "encoding/json"
  "github.com/gorilla/mux"
  "activity"
)

func main() {
  router := mux.NewRouter().StrictSlash(true)
//  router.HandleFunc("/", index_hdr)
//  router.HandleFunc("/register/{name}", register_hdr)
  for _, route := range global_routes {
    router.
          Methods(route.Method).
          Handler(route.HandlerFunc).
          Path(route.Pattern).
          Name(route.Name)
  }
  log.Fatal(http.ListenAndServe(":7777", router))
}

type Route struct {
  Name string
  Method string
  Pattern string
  HandlerFunc http.HandlerFunc
}

type Routes []Route

var global_routes = Routes {
  Route{
    "index",
    "GET",
    "/",
    index_hdr,
  },
  Route{
    "register",
    "POST",
    "/register/{name}",
    register_hdr,
  },
}

func index_hdr(w http.ResponseWriter, r *http.Request) {
  fmt.Fprintf(w, "Hello, %q", html.EscapeString(r.URL.Path))
}

func register_hdr(w http.ResponseWriter, r *http.Request) {
  vars := mux.Vars(r)
  fmt.Fprintf(w, "[register] name: %q\n", html.EscapeString(vars["name"]))

  partners := activity.Partners {
    activity.Partner{Name: vars["name"], Created: time.Now()},
  }

  err := activity.Insert(partners)
  if err == nil {
    json.NewEncoder(w).Encode(partners)
  }
}

