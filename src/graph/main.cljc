(ns graph.main
  (:require [hyperfiddle.electric :as e]
            [hyperfiddle.electric-dom2 :as dom]))

(e/defn Main [ring-request]
  (e/client
   (let [c (e/client e/system-time-ms)
         s (e/server e/system-time-ms)]
     (binding [dom/node js/document.body]
       (dom/div (dom/text "client time: " c))
       (dom/div (dom/text "server time: " s))
       (dom/div (dom/text "difference: " (- s c)))))))