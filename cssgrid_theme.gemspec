# coding: utf-8

Gem::Specification.new do |spec|
  spec.name          = "cssgrid_theme"
  spec.version       = "0.1.0"
  spec.authors       = ["Pritam Sukumar"]
  spec.email         = ["pritamps@gmail.com"]

  spec.summary       = "Nice sweet summary"
  spec.homepage      = "TODO: Put your gem's website or public repo URL here."
  spec.license       = "MIT"

  spec.files = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^(_(includes|layouts|sass)/|(LICENSE|README)((\.(txt|md|markdown)|$)))}i)
  end
  
  spec.add_runtime_dependency "jekyll", "~> 3.5"

  spec.add_development_dependency "bundler", "~> 1.12"
  spec.add_development_dependency "rake", "~> 10.0"
end
