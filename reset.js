module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "app"
    }
  }, {
    method: "fs.rm",
    params: {
      path: "workflows"
    }
  }]
}
