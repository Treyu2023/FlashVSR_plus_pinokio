module.exports = {
  run: [
    {
      // Drop models and outputs junctions before fs.rm.
      // rmdir removes the link. It does not delete O:\MODELS or D:\OUTPUTS\FlashVSR.
      method: "shell.run",
      params: {
        message: [
          "if exist app\\models rmdir app\\models",
          "if exist app\\outputs rmdir app\\outputs"
        ]
      }
    },
    {
      method: "fs.rm",
      params: {
        path: "app"
      }
    }
  ]
}
