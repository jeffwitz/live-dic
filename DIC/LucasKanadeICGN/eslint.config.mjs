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
    linterOptions: { reportUnusedDisableDirectives: "error" },
    rules: {
      "no-async-promise-executor": "error",
      "no-promise-executor-return": "error",
      "no-unsafe-finally": "error",
      "no-dupe-args": "error",
      "no-dupe-class-members": "error",
      "no-constant-condition": "error",
      "no-unexpected-multiline": "error",
      "no-unsafe-negation": "error",
      "no-constant-binary-expression": "error",
      "no-dupe-keys": "error",
      "no-undef": "error",
      "no-unreachable": "error",
      "no-unused-vars": [
        "error",
        { args: "none", caughtErrors: "none", varsIgnorePattern: "^_" },
      ],
      "valid-typeof": "error",
      "eqeqeq": ["error", "smart"],
      "for-direction": "error",
      "no-self-compare": "error",
      "no-self-assign": "error",
      "no-duplicate-case": "error",
      "use-isnan": "error",
    },
  },
  {
    files: ["python/src/lucas_kanade_icgn/web/core/**/*.js"],
    rules: {
      // The same core runs in a page and in a dedicated Worker.
      "no-restricted-globals": ["error", "window", "document"],
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
