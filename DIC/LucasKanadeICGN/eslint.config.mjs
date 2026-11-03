import globals from "globals";

const firstPartyJavaScript = [
  "web-app/**/*.js",
  "python/src/lucas_kanade_icgn/web/**/*.js",
  "three/LucasKanadeICGNThree.js",
];

export default [
  {
    ignores: ["**/vendor/**", "benchmarks/results/**"],
  },
  {
    files: firstPartyJavaScript,
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.browser,
        ...globals.worker,
        GPUBufferUsage: "readonly",
        GPUMapMode: "readonly",
        GPUShaderStage: "readonly",
        GPUTextureUsage: "readonly",
      },
    },
    rules: {
      "no-dupe-keys": "error",
      "no-undef": "error",
      "no-unreachable": "error",
      "no-unused-vars": [
        "error",
        { args: "none", caughtErrors: "none", varsIgnorePattern: "^_" },
      ],
      "valid-typeof": "error",
    },
  },
  {
    files: ["web-app/sw.js"],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.serviceworker,
        __PRECACHE__: "readonly",
      },
    },
  },
];
