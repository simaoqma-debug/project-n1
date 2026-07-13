{
  description = "Pocket Assistant Flask API and Astro/SolidJS frontend";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      projectFor =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          openai = pkgs.python312Packages.openai.overridePythonAttrs (_previous: {
            doCheck = false;
            nativeCheckInputs = [ ];
          });
          pythonEnvironment = python.withPackages (
            pythonPackages: with pythonPackages; [
              flask
              openai
            ]
          );
          frontend = pkgs.buildNpmPackage {
            pname = "pocket-assistant-frontend";
            version = "0.1.0";
            src = ./frontend;
            nodejs = pkgs.nodejs_24;
            npmDepsHash = "sha256-quGJjeqJe/AWAYK10E8Cwyaw+TIceu+7Dk+ky+kNE1g=";
            npmBuildScript = "build";
            ASTRO_TELEMETRY_DISABLED = "1";
            installPhase = ''
              runHook preInstall
              mkdir -p "$out"
              cp -r dist/. "$out/"
              runHook postInstall
            '';
          };
          runner = pkgs.writeShellApplication {
            name = "pocket-assistant";
            runtimeInputs = [ pythonEnvironment ];
            text = ''
              if [[ -z "''${OPENAI_API_KEY:-}" ]]; then
                echo "Aviso: defina OPENAI_API_KEY antes de utilizar a voz." >&2
              fi
              export POCKET_ASSISTANT_FRONTEND_DIST=${frontend}
              export PYTHONPATH=${self}
              exec ${pythonEnvironment}/bin/python -m server
            '';
          };
        in
        {
          inherit frontend pythonEnvironment runner;
          inherit pkgs;
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          project = projectFor system;
        in
        {
          default = project.runner;
          frontend = project.frontend;
        }
      );

      apps = forAllSystems (
        system:
        let
          project = projectFor system;
        in
        {
          default = {
            type = "app";
            program = "${project.runner}/bin/pocket-assistant";
          };
        }
      );

      devShells = forAllSystems (
        system:
        let
          project = projectFor system;
        in
        {
          default = project.pkgs.mkShell {
            packages = [
              project.pythonEnvironment
              project.pkgs.nodejs_24
              project.pkgs.uv
            ];
          };
        }
      );
    };
}
