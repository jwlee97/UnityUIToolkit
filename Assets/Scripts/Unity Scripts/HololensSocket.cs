using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

#if !UNITY_EDITOR
    using System.Threading.Tasks;
    using Windows.Networking;
    using Windows.Networking.Sockets;
    using Windows.Storage.Streams;
#endif

//Able to act as a reciever 
public class HololensSocket : MonoBehaviour {

#if !UNITY_EDITOR
    StreamSocket socket;
    StreamSocketListener listener;
    StreamSocketListenerConnectionReceivedEventArgs args;
    String port;
    String message;
#endif

    // Use this for initialization
    void Start() {
#if !UNITY_EDITOR
        listener = new StreamSocketListener();
        port = "9090";
        listener.ConnectionReceived += Listener_ConnectionReceived;
        listener.Control.KeepAlive = false;

        Listener_Start();
#endif
    }

#if !UNITY_EDITOR
    private async void Listener_Start() {
        Debug.Log("Listener started!");
        try {
            await listener.BindServiceNameAsync(port);
        } catch (Exception e) {
            Debug.Log("Error: " + e.Message);
            print("Error: " + e.Message);
        }

        Debug.Log("Listening for connection...");
    }

    private async void Listener_ConnectionReceived(StreamSocketListener sender, StreamSocketListenerConnectionReceivedEventArgs a) {
        Debug.Log("Listening for connection...");

        try {
            args = a;
            print("Connection established!");
        } catch (Exception e ) {
            Debug.Log("Disconnected! " + e);
        }
    }

    public void sendImageData(Serialization.ImageObject imageObject) {
        sendData(imageObject.imageDataBase64, imageObject.c2wM, imageObject.projM);
    }
    

    public async Task sendData(string imgData, string c2wM, string projM) {
        DataWriter writer;

        // Create the data writer object backed by the in-memory stream. 
        using (writer = new DataWriter(this.args.Socket.OutputStream)) {
            // Set the Unicode character encoding for the output stream
            writer.UnicodeEncoding = Windows.Storage.Streams.UnicodeEncoding.Utf8;
            // Specify the byte order of a stream.
            writer.ByteOrder = Windows.Storage.Streams.ByteOrder.LittleEndian;

            // Gets the size of UTF-8 string.
            string message = imgData + "/ffff/" + c2wM + ";" + projM;
            writer.MeasureString(message);

            // Write a string value to the output stream.
            writer.WriteString(message);

            // Send the contents of the writer to the backing stream.
            try {
                await writer.StoreAsync();
            }  catch (Exception exception) {
                switch (SocketError.GetStatus(exception.HResult))
                {
                    case SocketErrorStatus.HostNotFound:
                        // Handle HostNotFound Error
                        Debug.Log("HostNotFound Error in sendData");
                        throw;
                    default:
                        // If this is an unknown status it means that the error is fatal and retry will likely fail.
                        Debug.Log("Unknown SocketError in sendData");
                        throw;
                }
            }

            await writer.FlushAsync();
            // In order to prolong the lifetime of the stream, detach it from the DataWriter
            writer.DetachStream();
            writer.Dispose();
        }
    }
#endif

}