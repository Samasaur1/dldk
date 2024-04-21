{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs = { self, nixpkgs}:
    let
      allSystems = nixpkgs.lib.systems.flakeExposed;
      forAllSystems = nixpkgs.lib.genAttrs allSystems;
      define = f: forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
        in
          f pkgs
      );
      autorom' = pkgs: myPy:
        let
          autorom-without-roms = myPy.pkgs.buildPythonPackage rec {
            pname = "autorom";
            version = "0.6.1";
            pyproject = true;

            src = pkgs.fetchPypi {
              pname = "AutoROM";
              inherit version;
              hash = "sha256-bv8fG5ap1RlXdDf3HZao07iWI47KNDOo5pxckvbeMjE=";
            };

            propagatedBuildInputs = [
              myPy.pkgs.farama-notifications
              myPy.pkgs.requests
              myPy.pkgs.tqdm
              myPy.pkgs.click
            ];

            build-system = [
              myPy.pkgs.setuptools
              myPy.pkgs.wheel
            ];

            doCheck = false;

            meta = {
              homepage = "https://github.com/Farama-Foundation/AutoROM";
            };
          };
          romfile = pkgs.fetchurl {
            url = "https://gist.githubusercontent.com/jjshoots/61b22aefce4456920ba99f2c36906eda/raw/00046ac3403768bfe45857610a3d333b8e35e026/Roms.tar.gz.b64";
            hash = "sha256-Asp3fBZHanL6NmgKK6ePJMOsMbIVUDNUml83oGUxF94=";
          };
          romfile-targz = pkgs.runCommandNoCC "Roms.tar.gz" {} ''
            ${pkgs.coreutils}/bin/base64 --decode ${romfile} > $out
          '';
          roms = pkgs.stdenv.mkDerivation {
            name = "atari-roms";

            buildCommand = ''
              mkdir -p $out/${myPy.sitePackages}/AutoROM/roms
              ${autorom-without-roms}/bin/AutoROM -y -d $out/${myPy.sitePackages}/AutoROM/roms -s ${romfile-targz}
            '';
          };
          autorom = pkgs.symlinkJoin {
            name = "autorom";
            paths = [
              autorom-without-roms
              roms
            ];
          };
        in
          autorom;
      pyDeps = pkgs: pyPkgs:
        let
          shimmy = pyPkgs.buildPythonPackage rec {
            pname = "shimmy";
            version = "1.3.0";
            pyproject = true;

            src = pkgs.fetchPypi {
              pname = "Shimmy";
              inherit version;
              hash = "sha256-9F++qoGg51WryCUdV0HNS31d3dADqszaeWDmK+6CtJM=";
            };

            propagatedBuildInputs = [
              pyPkgs.numpy
              pyPkgs.gymnasium
            ];

            build-system = [
              pyPkgs.setuptools
              pyPkgs.wheel
            ];

            doCheck = false;

            meta = {
              homepage = "https://github.com/Farama-Foundation/Shimmy";
            };
          };
          ale-py-bin = pyPkgs.buildPythonPackage {
            version = "0.8.1";

            pname = "ale-py";

            format = "wheel";

            src = pkgs.fetchurl {
              url = "https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/download/v0.8.1/ale_py-0.8.1-cp311-cp311-macosx_11_0_arm64.whl";
              hash = "sha256-8Qsd+HdLvjsANldIteDgfPNfanA7uv+ZG8ezsiR9zMk=";
            };

            propagatedBuildInputs = [
              pyPkgs.typing-extensions
              pyPkgs.importlib-resources
              pyPkgs.numpy
            ];

            dontStrip = true;

            pythonImportsCheck = [ "ale_py" ];
          };
        in
          with pyPkgs; [
              jupyter
              numpy
              matplotlib
              tensorboard
              scikit-learn
              tqdm
              gymnasium
              seaborn
              pygame
              shimmy
          ] ++ (if (pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64) then with pyPkgs; [
              torch-bin
              torchvision-bin
              ale-py-bin
          ] else with pyPkgs; [
              torch
              torchvision
              ale-py
              pybox2d
          ]);
    in
    {
      packages = define (pkgs:
      let
        myPy = pkgs.python3;
      in
      {
        default = pkgs.python3.pkgs.buildPythonApplication {
          pname = "dldk";
          version = "0.1.0";
          pyproject = true;
          src = ./.;
          dependencies = (pyDeps pkgs myPy.pkgs) ++ [ (autorom' pkgs myPy) ];
          build-system = [ pkgs.python3.pkgs.poetry-core ];
        };
      });
      devShells = define (pkgs:
      let
        myPy = pkgs.python3;
      in
      {
        default = pkgs.mkShell {
            buildInputs = [
              (myPy.withPackages (pyDeps pkgs))
              (autorom' pkgs myPy)
            ];
          };
        }
      );
      apps = define (pkgs: {
        interactive = {
          type = "app";
          program = "${self.packages.${pkgs.system}.default}/bin/interactive";
        };
        train = {
          type = "app";
          program = "${self.packages.${pkgs.system}.default}/bin/train";
        };
        showoff = {
          type = "app";
          program = "${self.packages.${pkgs.system}.default}/bin/showoff";
        };
      });
    };
}
