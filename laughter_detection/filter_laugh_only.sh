DIR=$1
#/group/corpora/public/icsi_meeting/transcripts/transcripts/'

NO_TEXT="text()[normalize-space()='']"

sum=0


for file in ${DIR}/*.mrt; do
    echo $file
    count=$(xmllint --xpath "count(//Segment[VocalSound[contains(@Description,'laugh')][preceding-sibling::$NO_TEXT and following-sibling::$NO_TEXT] and count(./*) < 2])" $file)
    sum=$(( $sum + $count ))
done;

echo $sum
