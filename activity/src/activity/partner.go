package activity

import (
  "time"
)

type Partner struct {
  Name string `json:"name"`
  Level int   `json:"level"`
  Created time.Time `json:"created"`
}

type Partners []Partner

