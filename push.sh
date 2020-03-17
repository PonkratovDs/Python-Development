#!/bin/bash
read -t 100 -p "Enter comment, please: " comment

if [ -n "$comment" ]; then
    git add -A && git commit -m "$comment" && git push
else    
    echo
    echo "No comment"
fi