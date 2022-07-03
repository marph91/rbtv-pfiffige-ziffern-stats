# RBTV Pfiffige Ziffern Stats

Datensammlung und Visualisierungen für die Show Pfiffige Ziffern auf Rocketbeans TV.

Webseite: <https://marph91.github.io/rbtv-pfiffige-ziffern-stats/>

## Mitmachen

Es gibt einige Stellen, an denen geholfen werden kann:

- Daten für neue Episoden hinzufügen: Dazu einfach die Schätzungen in `data/guesses.csv` und die Metadaten in `data/episodes.csv` ergänzen.
- Neue Auswertungen ergänzen: Dazu am Besten als erstes ein [Issue](https://github.com/marph91/rbtv-pfiffige-ziffern-stats/issues) erstellen, in dem die Implementierung diskutiert werden kann.
- Review der Grafiken und Tabellen: Die Daten wurden alle manuell eingegeben. Auch die Implementierung kann fehlerhaft sein. Falls dir etwas auffällt, erstelle gern ein [Issue](https://github.com/marph91/rbtv-pfiffige-ziffern-stats/issues).

## Workflow

Die Seite <https://marph91.github.io/rbtv-pfiffige-ziffern-stats/> wird bei jedem Commit automatisiert gebaut. Der Ablauf ist in der [Build-Action](.github/workflows/build.yml) definiert:

1. Aktualisieren der Daten in `data/*.csv`, der Implementierung in `scripts/*.py` oder der Templates in `templates/*.md.j2`.
2. Ausführen des Skripts `scripts/generate_charts_tables.py`, das die Grafiken erstellt und die Templates rendert.
3. Deployen der Seite mittels `mkdocs gh-deploy`. Anschließend sollte die Seite mit etwas Verzögerung aktualisiert sein.
