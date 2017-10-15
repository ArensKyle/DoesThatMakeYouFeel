entries=$(find data/. -type f)
cd twitter_download/

for i in $entries; do
    if [[ $i =~ twitter.*\.txt$ ]]; then
        echo "downloading ${i}..."
        python download_tweets_api.py --dist=$i --output="$i.download"
    fi
done
