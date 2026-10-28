from pathlib import Path
root=Path(__file__).resolve().parent
source=root/'index_MarketPort_V7.html'
target=root/'index.html'
if not source.exists(): raise SystemExit('Placez index_MarketPort_V7.html à côté de ce script.')
if not (root/'app.py').exists(): raise SystemExit('app.py absent : placez les fichiers dans le dossier du projet.')
if target.exists():
    backup=root/'index.html.avant-v7.bak'
    if not backup.exists(): backup.write_bytes(target.read_bytes())
target.write_bytes(source.read_bytes())
print('V7 installée. app.py inchangé. Ne relancez pas les scripts V4/V5/V6 après cette installation.')
