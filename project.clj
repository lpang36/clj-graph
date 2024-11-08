(defproject graph "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [incanter "1.9.3"]
                 [org.clojure/tools.trace "0.8.0"]
                 [org.clojure/data.json "2.5.0"]]
  :main ^:skip-aot graph.train
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}}
  :dev-dependencies [[lein-git-deps "0.0.1-SNAPSHOT"]]
  :git-dependencies [["https://github.com:hyperfiddle/electric.git"]])