{
  description = "AeroMDO Copilot - Surrogate-Assisted MDO & AI Engineering Assistant for SAE AeroDesign";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        numpy
        scipy
        pandas
        openpyxl
        scikit-learn
        xgboost
        fastapi
        uvicorn
        pydantic
        pytest
        httpx
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pythonEnv
          pkgs.poppler-utils
          pkgs.sqlite
          pkgs.jq
        ];

        shellHook = ''
          export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
          echo "AeroMDO Copilot devShell activated (Python $(python3 --version))"
        '';
      };
    };
}
