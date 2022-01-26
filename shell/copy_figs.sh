(cd bilateral-connectome/overleaf && git pull)
rsync -r --max-size=49m ./bilateral-connectome/results/figs ./bilateral-connectome/overleaf
(cd bilateral-connectome/overleaf && git add . && git commit -m 'update figures' && git push)