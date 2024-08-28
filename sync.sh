#!/bin/bash
rsync -arv --exclude={.git,Validation,Logs,Analysis} . jamie@100.69.105.112:~/code/
rsync -avh jamie@100.69.105.112:~/code/Logs .
rsync -avh jamie@100.69.105.112:~/code/Validation .
rsync -avh jamie@100.69.105.112:~/code/Analysis .