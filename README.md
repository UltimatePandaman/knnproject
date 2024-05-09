<!--- Autoři: Zbyněk Lička, Martin Bublavý, Daniel Prudký -->

Projekt do předmětu KNN. Založený na implementaci StarGANv2-VC [implementaci StarGANv2-VC]([https://website-name.com](https://github.com/yl4579/StarGANv2-VC)).

Projekt má sloužit pro kvalitní změnu hlasu mezi dvěma řečníky. Implementačně se zaměřuje na využití StarGANv2-VC technologií pro CycleGAN architekturu.

Inspirováno články:
[StarGANv2-VC](https://arxiv.org/abs/2107.10394)
[CycleGAN-VC3](https://arxiv.org/abs/2010.11672)

### Instalace
```bash
# Vytvořte venv a aktivujte jej
pip3 install -r requirements.txt

get_dataset.sh
get_models.sh
get_vocoder.sh
```

### Použití
#### Sady dat
Vytvořte trénovací, validační a testovací datalisty pomocí `prepare_train_list.ipynb`. Můžete si zvolit *dvojici* speakerů v proměnné `speakers`.

#### Trénování
Trénovací parametry lze pozměnit v `Configs/config.yml`. Doporučuji měnit pouze `device`, `batch_size`, `epochs` atd. Váhy ztrátových funkcí jsou jakž takž optimalizované. Jakmile si zvolíte své vysněné parametry, stačí spustit:
```bash
python3 train.py
```
Trénování pro všechny ztrátové funkce trvá cca 12 hodin na 10 GB GPU s 16 GB RAM a 1 CPU. Pro tento projekt je nutno alokovat velké množství paměti (15-50 GB) záleží na množství natrénovaných modelů

#### Inference
Inference se provádí pomocí `inference.ipynb`.


### Soubory
Přidané:
+ get_dataset.sh stažení a umístění datasetu
+ get_models.sh stažení a umístění ASR a F0 předtrénovaných sítí
+ get_vocoder.sh stažení a umístění předtrénovaného vocoderu
+ run.sh PBS skript
+ LICENSE naše licence
+ requirements.txt

Modifikované:
+ Data/VCTK.ipynb přejmenováno na prepare_train_list.ipynb a přemístěno do hlavního adresáře, modifikováno, aby produkoval dvojice cest k nahrávkám (trénovacím, validačním, testovacím)
+ Demo/inference.ipynb přemístěno do hlavního adresáře, rozsáhle upraveno pro funkci v CycleGAN architektuře
+ losses.py všechny ztrátové funkce se počítají 2x, přidána identity loss, second-step adversarial loss, odebrány style loss funkce
+ meldataset.py dataset nyní vrací dvojice mel-spectrogramů, jeden ukázkový pro každého řečníka
+ models.py odebrány všechny zmínky o style encoder, Generator upraven, aby pracoval bez style encoder, součástí tohoto bylo předělání AdaINResBlk na ResBlkDecoder, odebrán mapping network
+ trainer.py odebrán mapping network learning rate, upraveno jméno uloženého checkpointu modelu
+ train.py 
+ LICENSE StarGANv2-VC licence přesunuta do licenses/
+ transforms.py jenom přidány komentáře
+ optimizers.py jenom přidány komentáře

Odebráno
- README.md nahrazeno za náš vlastní readme.md
