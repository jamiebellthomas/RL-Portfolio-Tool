#!/bin/bash
rsync -arv --exclude={.git,Validation,Logs} . jamie@100.69.105.112:~/code/
rsync -avh jamie@100.69.105.112:~/code/Logs .