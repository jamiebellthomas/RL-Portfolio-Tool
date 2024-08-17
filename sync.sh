#!/bin/bash
rsync -avh --exclude .git . jamie@100.69.105.112:~/code/
rsync -avh jamie@100.69.105.112:~/code/Logs .