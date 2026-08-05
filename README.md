# PDF Grid Engine

## Vision

PDF Grid Engine est une nouvelle implémentation expérimentale d'un moteur de Grid Trading inspiré du modèle patrimonial décrit dans le document de recherche de référence.

Ce projet n'est pas une évolution de **ton_grid_bot**.

Il s'agit d'un moteur entièrement nouveau, conçu à partir d'une feuille blanche, dont l'objectif est de reproduire le plus fidèlement possible la logique économique du modèle théorique avant toute adaptation au monde réel.

---

## Philosophie

Le développement suit une démarche inverse de celle d'un développement logiciel classique.

La théorie est définie avant l'implémentation.

Le code devra toujours respecter les documents d'architecture.

Les décisions d'implémentation ne devront jamais modifier les principes du modèle théorique.

---

## Documents fondateurs

Le moteur est actuellement défini par les notes de recherche :

- RN-100 — Modèle patrimonial
- RN-101 — Théorie des deux mondes
- RN-102 — Théorie de l'écart

Ces documents constituent la spécification de référence.

---

## Objectifs

Construire un moteur :

- fidèle au modèle patrimonial ;
- indépendant de toute notion de lot ou de FIFO ;
- séparant strictement le monde théorique du monde réel ;
- capable de fonctionner avec plusieurs adaptateurs d'exchange.

---

## État du projet

Le projet est actuellement en phase de conception.

Aucun composant métier n'est encore implémenté.

La priorité est de stabiliser l'architecture avant toute écriture de code.

---

## Dépôt historique

Le développement du moteur actuellement en production continue dans le dépôt :

ton_grid_bot

Les deux projets évolueront indépendamment.
