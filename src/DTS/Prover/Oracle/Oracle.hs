{-# LANGUAGE OverloadedStrings #-}
module DTS.Prover.Oracle.Oracle (oracle, oracleBuilder) where

import qualified Data.List as L 
import qualified DTS.DTTdeBruijn as DdB
import qualified Data.Map.Strict as M
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as TIO
import qualified Data.Text.Lazy.Read as TR
import Data.Maybe (mapMaybe)
import DTS.Prover.Oracle.OracleUtils (loadWordEmbeddings, calcPScore)

oracle :: (DdB.ConName, DdB.ConName) -> IO Float
oracle (childName, parentName) = do
    let embPath = "src/DTS/Prover/Oracle/Embedding.csv"
    let alpha   = 0.1

    embMap <- loadWordEmbeddings embPath

    let pResult = M.lookup parentName embMap
    let cResult = M.lookup childName embMap

    case (pResult, cResult) of
        (Just pEmbs, Just cEmbs) -> do
            let allScores =
                    [ calcPScore alpha pEmb cEmb
                    | pEmb <- pEmbs
                    , cEmb <- cEmbs
                    ]
            if null allScores
                then return 0.0
                else return (maximum allScores)
        _ -> return 0.0

oracleBuilder :: IO (DdB.ConName -> DdB.ConName -> Float)
oracleBuilder = do
    let embPath = "src/DTS/Prover/Oracle/Embedding.csv"
    let alpha   = 0.1
    embMap <- loadWordEmbeddings embPath
    return $ \childName parentName ->
        let pResult = M.lookup parentName embMap
            cResult = M.lookup childName embMap
        in case (pResult, cResult) of
            (Just pEmbs, Just cEmbs) -> 
                let allScores = [ calcPScore alpha pEmb cEmb | pEmb <- pEmbs, cEmb <- cEmbs ]
                in if null allScores 
                   then 0.0 
                   else maximum allScores
            _ -> 0.0
