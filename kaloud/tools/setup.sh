targets="
gopkg.in/redis.v3
"

for p in $targets ;do
  go get $p
done

