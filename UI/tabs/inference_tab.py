from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QFileDialog, QProgressBar, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class ZoneDepotImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setText("⬆️  Déposer une image ou cliquer pour parcourir\nPNG, JPG — redimensionné automatiquement")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaaaaa;
                border-radius: 10px;
                padding: 10px;
                color: #888888;
                font-size: 11px;
            }
        """)
        self.setFixedHeight(70)

    def dragEnterEvent(self, evenement):
        if evenement.mimeData().hasUrls():
            evenement.acceptProposedAction()

    def dropEvent(self, evenement):
        urls = evenement.mimeData().urls()
        if urls:
            chemin = urls[0].toLocalFile()
            self.parent().charger_image(chemin)


class OngletInference(QWidget):
    def __init__(self):
        super().__init__()
        self.chemin_image = None
        self.historique = []

        disposition_principale = QHBoxLayout()
        disposition_principale.setSpacing(20)
        disposition_principale.setContentsMargins(20, 20, 20, 20)

        #GAUCHE
        widget_gauche = QWidget()
        disposition_gauche = QVBoxLayout()
        disposition_gauche.setSpacing(8)

        label_entree = QLabel("ENTRÉE")
        label_entree.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        disposition_gauche.addWidget(label_entree)

        # btn img / Doss
        disposition_mode = QHBoxLayout()
        self.btn_image_unique = QPushButton("Image unique")
        self.btn_dossier = QPushButton("Dossier batch")
        self.btn_image_unique.setCheckable(True)
        self.btn_image_unique.setChecked(True)
        self.btn_dossier.setCheckable(True)
        self.btn_image_unique.setStyleSheet("""
            QPushButton {                          
                padding: 8px;
                border-radius: 8px;
                background: #1592A2 ;
                color: white;
            }
            QPushButton:hover { background: #0e6b77; }
        """)
        self.btn_dossier.setStyleSheet("padding: 8px; border-radius: 8px;")
        self.btn_image_unique.clicked.connect(lambda: self.changer_mode("unique"))
        self.btn_dossier.clicked.connect(lambda: self.changer_mode("dossier"))
        disposition_mode.addWidget(self.btn_image_unique)
        disposition_mode.addWidget(self.btn_dossier)
        disposition_gauche.addLayout(disposition_mode)

        #depot img
        self.zone_depot = ZoneDepotImage(self)
        self.zone_depot.mousePressEvent = lambda e: self.parcourir_image()
        disposition_gauche.addWidget(self.zone_depot)

        # Apercu img
        self.apercu_image = QLabel()
        self.apercu_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.apercu_image.setFixedHeight(70)
        self.apercu_image.setStyleSheet("border: 1px solid #ccc; border-radius: 8px;")
        self.apercu_image.setText("28 × 28 px")
        disposition_gauche.addWidget(self.apercu_image)
        disposition_gauche.addSpacing(10)

        # norma
        disposition_normalisation = QHBoxLayout()
        label_normalisation = QLabel("Appliquer la normalisation du dataset")
        label_normalisation.setStyleSheet("font-size: 11px;")
        self.toggle_normalisation = QPushButton("●")
        self.toggle_normalisation.setCheckable(True)
        self.toggle_normalisation.setChecked(True)
        self.toggle_normalisation.setFixedSize(50, 25)
        self.toggle_normalisation.setStyleSheet("border-radius: 12px; background: #4CAF50; color: white;")
        disposition_normalisation.addWidget(label_normalisation)
        disposition_normalisation.addStretch()
        disposition_normalisation.addWidget(self.toggle_normalisation)
        disposition_gauche.addLayout(disposition_normalisation)

        description_normalisation = QLabel("Les stats de normalisation sont appliquées automatiquement.")
        description_normalisation.setStyleSheet("font-size: 10px; color: #888;")
        disposition_gauche.addWidget(description_normalisation)

        # btn Lancer l'infrc
        self.btn_lancer = QPushButton("Lancer l'inférence")
        self.btn_lancer.setStyleSheet("""
            QPushButton {
                padding: 10px;
                border-radius: 8px;
                background: #1592A2;
                color: white;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background: #0e6b77; }
        """)
        self.btn_lancer.clicked.connect(self.lancer_inference)
        disposition_gauche.addWidget(self.btn_lancer)

        disposition_gauche.addStretch()
        widget_gauche.setLayout(disposition_gauche)
        widget_gauche.setMaximumWidth(320)
        disposition_principale.addWidget(widget_gauche)

        # --- DROITE ---
        widget_droite = QWidget()
        disposition_droite = QVBoxLayout()
        disposition_droite.setSpacing(8)

        label_resultat = QLabel("RÉSULTAT")
        label_resultat.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        disposition_droite.addWidget(label_resultat)

        self.label_classe = QLabel("-- Aucune prédiction --")
        self.label_classe.setStyleSheet("""
            background: #1592A2;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: bold;
            color: white;
        """)
        self.label_classe.setAlignment(Qt.AlignmentFlag.AlignCenter)
        disposition_droite.addWidget(self.label_classe)

        disposition_stats = QHBoxLayout()
        disposition_stats.setSpacing(8)
        self.stat_prediction = self.creer_boite_stat("--", "indice classe", "Prédit")
        self.stat_confiance  = self.creer_boite_stat("--%", "top-1", "Confiance")
        self.stat_temps      = self.creer_boite_stat("--ms", "temps réel", "Inférence")
        disposition_stats.addWidget(self.stat_prediction)
        disposition_stats.addWidget(self.stat_confiance)
        disposition_stats.addWidget(self.stat_temps)
        disposition_droite.addLayout(disposition_stats)

        label_top5 = QLabel("Top-5 probabilités")
        label_top5.setStyleSheet("font-size: 12px; color: #888; margin-top: 6px;")
        disposition_droite.addWidget(label_top5)

        self.barres_probabilites = []
        for i in range(5):
            ligne = QHBoxLayout()
            nom = QLabel("--")
            nom.setFixedWidth(100)
            barre = QProgressBar()
            barre.setMaximum(100)
            barre.setValue(0)
            barre.setTextVisible(False)
            barre.setFixedHeight(8)
            barre.setStyleSheet("QProgressBar::chunk { background: #1592A2; border-radius: 4px; }")
            pourcentage = QLabel("0%")
            pourcentage.setFixedWidth(30)
            pourcentage.setAlignment(Qt.AlignmentFlag.AlignRight)
            ligne.addWidget(nom)
            ligne.addWidget(barre)
            ligne.addWidget(pourcentage)
            self.barres_probabilites.append((nom, barre, pourcentage))
            disposition_droite.addLayout(ligne)

        label_historique = QLabel("HISTORIQUE")
        label_historique.setStyleSheet("font-weight: bold; font-size: 10px; color: #ffffff; margin-top: 8px;")
        disposition_droite.addWidget(label_historique)

        self.label_historique = QLabel("Aucune prédiction encore")
        self.label_historique.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_historique.setStyleSheet("""
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            color: #ffffff;
            font-size: 11px;
        """)
        disposition_droite.addWidget(self.label_historique)

        disposition_droite.addStretch()
        widget_droite.setLayout(disposition_droite)
        disposition_principale.addWidget(widget_droite)

        self.setLayout(disposition_principale)

    #3 STAT
    def creer_boite_stat(self, valeur, sous_titre, titre):
        cadre = QFrame()
        cadre.setStyleSheet("border: 1px solid #ddd; border-radius: 8px; padding: 6px;")
        cadre.setMinimumWidth(120)
        disposition = QVBoxLayout()
        disposition.setSpacing(2)
        label_titre = QLabel(titre)
        label_titre.setStyleSheet("font-size: 11px; color: #888; border: none;")
        label_valeur = QLabel(valeur)
        label_valeur.setStyleSheet("font-size: 10px; font-weight: bold; border: none;")
        label_sous_titre = QLabel(sous_titre)
        label_sous_titre.setStyleSheet("font-size: 10px; color: #888; border: none;")
        disposition.addWidget(label_titre)
        disposition.addWidget(label_valeur)
        disposition.addWidget(label_sous_titre)
        cadre.setLayout(disposition)
        cadre.label_valeur = label_valeur
        return cadre

    def changer_mode(self, mode):
        if mode == "unique":
            self.btn_image_unique.setStyleSheet("""
                QPushButton {
                    padding: 8px;
                    border-radius: 8px;
                    background: #1592A2;
                    color: white;
                }
                QPushButton:hover { background: #0e6b77; }
            """)
            self.btn_dossier.setStyleSheet("padding: 8px; border-radius: 8px;")
        else:
            self.btn_dossier.setStyleSheet("""
                QPushButton {
                    padding: 8px;
                    border-radius: 8px;
                    background: #1592A2;
                    color: white;
                }
                QPushButton:hover { background: #0e6b77; }
            """)
            self.btn_image_unique.setStyleSheet("padding: 8px; border-radius: 8px;")

    def parcourir_image(self):
        chemin, _ = QFileDialog.getOpenFileName(self, "Choisir une image", "", "Images (*.png *.jpg *.jpeg)")
        if chemin:
            self.charger_image(chemin)

    def charger_image(self, chemin):
        self.chemin_image = chemin
        pixmap = QPixmap(chemin).scaled(150, 100, Qt.AspectRatioMode.KeepAspectRatio)
        self.apercu_image.setPixmap(pixmap)

    def lancer_inference(self):
        if not self.chemin_image:
            self.label_classe.setText("⚠️ Veuillez d'abord choisir une image !")
            return

        import random
        classes = ["T-shirt", "Pantalon", "Pull", "Robe", "Manteau",
                   "Sandale", "Chemise", "Basket", "Sac", "Bottine"]
        probabilites = sorted([random.random() for _ in range(10)], reverse=True)
        total = sum(probabilites)
        probabilites = [p / total * 100 for p in probabilites]

        classe_predite = classes[0]
        self.label_classe.setText(f"Classe 1 — {classe_predite}   {probabilites[0]:.1f}%")
        self.stat_prediction.label_valeur.setText("1")
        self.stat_confiance.label_valeur.setText(f"{probabilites[0]:.1f}%")
        self.stat_temps.label_valeur.setText("23ms")

        for i, (nom, barre, pourcentage) in enumerate(self.barres_probabilites):
            nom.setText(classes[i])
            barre.setValue(int(probabilites[i]))
            pourcentage.setText(f"{probabilites[i]:.1f}%")

        self.historique.append(f"{classe_predite} — {probabilites[0]:.1f}%")
        self.label_historique.setText("\n".join(self.historique[-5:]))