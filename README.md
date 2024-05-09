# Building

```bash
nix build .
```

# Running

## Interactive

```bash
./result/bin/interactive
```

## Training

```bash
./result/bin/train <strategy> [filename]
```
Leave `strategy` blank to see a list of options

## Running a trained model

```bash
./result/bin/train <strategy> [filename]
```
The default filename is `<strategy>.pt`, which is *not* the default filename produced by training

# Compiling the writeup

```bash
pandoc -t pdf --template pandoc.template --pdf-engine=pdflatex writeup.md -o 'Final Project — Sam Gauck.pdf'
```
