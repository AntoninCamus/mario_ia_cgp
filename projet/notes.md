### Interfaçage :
- https://pypi.org/project/gym-super-mario-bros/
- il existe aussi https://github.com/ppaquette/gym-super-mario (à tester) --tuto--> https://becominghuman.ai/getting-mario-back-into-the-gym-setting-up-super-mario-bros-in-openais-gym-8e39a96c1e41


### Analyse d'image :

- convolution -> | | -> (vecteur de param) -> entrée d'un perceptron -> réseau 
récurent (lstm)

- [ fenêtre de taille x ] * screenx * screenx
  -> Chaque surpixel est analysé pour en tirer les couleurs importantes

- Changer les textures des roms

### Prise de décision

- DQN
- NEAT
- Cartesian : https://arxiv.org/pdf/1806.05695.pdf

